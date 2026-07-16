"""
Modèle ELUS complet
--------------------
Assemble l'encodeur audio, l'encodeur texte et le module de matching.
Expose :
  - forward()          : pour l'entraînement (retourne les embeddings nécessaires aux pertes)
  - transcribe()        : inférence audio -> texte (recherche du plus proche voisin)
  - retrieve_audio()    : inférence texte -> audio (recherche inverse)
"""

import torch
import torch.nn as nn

from config import ELUSConfig
from audio_encoder import AudioEncoder
from text_encoder import TextEncoder
from matcher import build_matcher
from ctc_head import CTCHead


class ELUSModel(nn.Module):
    def __init__(self, cfg: ELUSConfig, ctc_num_classes: int = 0):
        super().__init__()
        self.cfg = cfg
        self.audio_encoder = AudioEncoder(cfg.audio)
        self.text_encoder = TextEncoder(cfg.text)
        self.matcher = build_matcher(cfg.matcher)
        self.ctc_head = None
        if ctc_num_classes > 0:
            self.ctc_head = CTCHead(cfg.audio.latent_dim, ctc_num_classes)

    def attach_ctc_head(self, num_classes: int, dropout: float = 0.1):
        self.ctc_head = CTCHead(self.cfg.audio.latent_dim, num_classes, dropout=dropout)
        return self.ctc_head

    def forward_audio_only(self, waveforms, audio_attention_mask):
        """Encodeur audio seul → séquence latente (pour CTC)."""
        return self.audio_encoder(waveforms, audio_attention_mask)

    def forward_ctc(self, waveforms, audio_attention_mask):
        """
        Audio → logits CTC (B, T, C).
        Utilise la séquence encodeur (pas le matcher).
        """
        if self.ctc_head is None:
            raise RuntimeError("Aucune tête CTC attachée (attach_ctc_head)")
        audio_out = self.audio_encoder(waveforms, audio_attention_mask)
        logits = self.ctc_head(audio_out["sequence_embeddings"])
        return logits, audio_out["seq_mask"]

    # ------------------------------------------------------------------ #
    # Entraînement contrastif
    # ------------------------------------------------------------------ #
    def forward(self, waveforms, audio_attention_mask, token_ids, text_attention_mask):
        audio_out = self.audio_encoder(waveforms, audio_attention_mask)
        text_out = self.text_encoder(token_ids, text_attention_mask)

        # Séquence brute (avant matcher) — utilisée pour la loss segmentale caractères.
        audio_out["sequence_raw"] = audio_out["sequence_embeddings"]

        # Le "pont" f(V_a) -> espace texte est appliqué à la séquence audio.
        # En mode "attention", chaque segment audio interroge les unités texte
        # de la MÊME phrase pendant l'entraînement (self-supervision guidée).
        projected_audio_seq = self.matcher(
            audio_out["sequence_embeddings"],
            text_out["unit_embeddings"],
            text_key_padding_mask=~text_out["unit_mask"],
        )
        audio_out["sequence_embeddings"] = projected_audio_seq

        # Le vecteur global audio est également reprojeté (moyenne pondérée
        # de la séquence déjà projetée) pour la perte globale.
        mask = audio_out["seq_mask"].unsqueeze(-1)
        pooled = (projected_audio_seq * mask).sum(1) / mask.sum(1).clamp(min=1)
        audio_out["global_embedding"] = torch.nn.functional.normalize(pooled, dim=-1)

        return audio_out, text_out

    # ------------------------------------------------------------------ #
    # Inférence : Audio -> Texte (transcription)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor, unit_vocab_embeddings: torch.Tensor,
                    unit_vocab_symbols: list, attention_mask: torch.Tensor = None,
                    topk: int = 1):
        """
        waveform : (1, num_samples) un seul énoncé (peut être une lettre isolée).
        unit_vocab_embeddings : (V, D) embeddings de TOUTES les unités du
            vocabulaire (précalculés une fois via `self.text_encoder` sur
            chaque symbole seul -- voir `build_unit_vocab_embeddings`).
        unit_vocab_symbols : liste des symboles correspondant aux lignes de
            `unit_vocab_embeddings` (ex: ["a", "b", ..., "ɖ", "ƒ", ...] pour l'éwé).

        Retourne la séquence de symboles décodés (un par segment audio),
        + les scores de confiance (similarité cosinus).
        """
        self.eval()
        audio_out = self.audio_encoder(waveform, attention_mask)
        seq = audio_out["sequence_embeddings"]  # (1, T, D)

        # Sans texte de référence disponible en inférence pure, le matcher
        # en mode MLP est utilisé (ou le matcher "attention" avec le lexique
        # entier comme key/value -- cf. `unit_vocab_embeddings`).
        vocab = unit_vocab_embeddings.unsqueeze(0)  # (1, V, D)
        projected = self.matcher(seq, vocab)  # (1, T, D)

        sims = projected.squeeze(0) @ unit_vocab_embeddings.t()  # (T, V)
        scores, indices = sims.topk(topk, dim=-1)

        decoded = [[unit_vocab_symbols[i] for i in row] for row in indices.tolist()]
        return decoded, scores

    @torch.no_grad()
    def transcribe_phrase(
        self,
        waveform: torch.Tensor,
        unit_vocab_embeddings: torch.Tensor,
        unit_vocab_symbols: list,
        attention_mask: torch.Tensor = None,
        level: str = "word",
        min_score: float = 0.15,
        refine_iters: int = 3,
        use_dtw: bool = True,
    ) -> tuple:
        """
        Audio -> phrase par détection d'unités dans l'espace latent.

        1) DTW monotone (ou argmax) frame -> unités caractères/mots
        2) Raffinement via matcher cross-attention sur l'hypothèse courte
        """
        from align import dtw_decode_units, dtw_decode_known_units
        from data import unit_display, unit_type

        self.eval()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        audio_out = self.audio_encoder(waveform, attention_mask)
        seq = audio_out["sequence_embeddings"][0, audio_out["seq_mask"][0].bool()]
        if seq.size(0) == 0:
            return "", [], []

        if use_dtw:
            unit_ids, unit_scores = dtw_decode_units(
                seq, unit_vocab_embeddings, min_score=min_score,
            )
        else:
            sims = seq @ unit_vocab_embeddings.t()
            pred_ids = sims.argmax(1).tolist()
            pred_scores = sims.max(1).values.tolist()
            runs = []
            for idx, sc in zip(pred_ids, pred_scores):
                if sc < min_score:
                    continue
                if not runs or runs[-1][0] != idx:
                    runs.append([idx, sc, 1])
                else:
                    runs[-1][1] += sc
                    runs[-1][2] += 1
            if not runs:
                return "", [], []
            unit_ids = [r[0] for r in runs]
            unit_scores = [r[1] / r[2] for r in runs]

        if not unit_ids:
            return "", [], []

        hyp_emb = unit_vocab_embeddings[unit_ids].unsqueeze(0)
        for _ in range(refine_iters):
            projected = self.matcher(seq.unsqueeze(0), hyp_emb)[0]
            sims2 = projected @ unit_vocab_embeddings.t()
            if use_dtw:
                new_ids, new_scores = dtw_decode_known_units(
                    projected, unit_vocab_embeddings[unit_ids],
                    unit_ids, min_score=min_score,
                )
            else:
                new_ids, new_scores = [], []
                chunk = max(1, projected.size(0) // len(unit_ids))
                for i, uid in enumerate(unit_ids):
                    s = i * chunk
                    e = projected.size(0) if i == len(unit_ids) - 1 else (i + 1) * chunk
                    score, idx = sims2[s:e].mean(0).max(0)
                    if score.item() >= min_score:
                        new_ids.append(idx.item())
                        new_scores.append(score.item())
            if new_ids:
                unit_ids = new_ids
                unit_scores = new_scores
                hyp_emb = unit_vocab_embeddings[unit_ids].unsqueeze(0)

        units = [unit_display(unit_vocab_symbols[i]) for i in unit_ids]
        if level == "word":
            phrase = " ".join(units)
        else:
            phrase = "".join(units)
        return phrase, units, unit_scores

    @torch.no_grad()
    def transcribe_from_corpus(
        self,
        waveform: torch.Tensor,
        corpus_texts: list,
        symbol_to_id: dict,
        attention_mask: torch.Tensor = None,
        shortlist: int = 40,
    ) -> tuple:
        """
        Audio -> phrase en cherchant la meilleure correspondance dans un corpus connu.
        Utile quand les phrases possibles sont dans un lexique fermé.
        """
        from data import tokenize_text

        self.eval()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(waveform, dtype=torch.long)

        # Pré-calculer embeddings texte global
        text_globals = []
        for t in corpus_texts:
            ids = tokenize_text(t, symbol_to_id, mode="words")
            if not ids:
                text_globals.append(None)
                continue
            q = torch.tensor([ids], device=waveform.device)
            m = torch.ones_like(q)
            out = self.text_encoder(q, m)
            text_globals.append(out["global_embedding"])

        # Shortlist via audio seul
        ao = self.audio_encoder(waveform, attention_mask)
        mask = ao["seq_mask"].unsqueeze(-1)
        a_q = torch.nn.functional.normalize(
            (ao["sequence_embeddings"] * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1
        )
        valid = [(i, tg) for i, tg in enumerate(text_globals) if tg is not None]
        if not valid:
            return "", -1.0
        idxs, embs = zip(*valid)
        text_db = torch.cat(list(embs), dim=0)
        sims = a_q @ text_db.t()
        k = min(shortlist, sims.size(1))
        _, top_local = sims.squeeze(0).topk(k)

        # Rerank : forward pairé audio + candidat (comme à l'entraînement)
        best_i, best_s = -1, -1.0
        for li in top_local.tolist():
            ci = idxs[li]
            t = corpus_texts[ci]
            ids = tokenize_text(t, symbol_to_id, mode="words")
            tok = torch.tensor([ids], device=waveform.device)
            tmask = torch.ones_like(tok)
            batch_w = waveform
            batch_m = attention_mask
            audio_out, text_out = self(batch_w, batch_m, tok, tmask)
            s = (audio_out["global_embedding"] * text_out["global_embedding"]).sum().item()
            if s > best_s:
                best_s, best_i = s, ci

        return corpus_texts[best_i], best_s

    # ------------------------------------------------------------------ #
    # Inférence : recherche par unité (mot, lettre, phrase)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def search_units(self, query: str, symbol_to_id: dict,
                     unit_types: dict = None, level: str = "all",
                     topk: int = 5, device: str = "cpu"):
        """
        Recherche les unités les plus proches d'une requête texte dans l'espace latent.
        level: 'word' | 'char' | 'all'
          - word  : cherche un mot isolé (ex: "ame")
          - char  : cherche une lettre isolée (ex: "ɖ")
          - all   : phrase entière (retrieval global via embedding CLS)
        Retourne liste de (unité, score, type).
        """
        self.eval()
        unit_types = unit_types or {k: ("word" if k.startswith("w:") else "char") for k in symbol_to_id}

        if level == "all":
            from data import tokenize_text
            ids = tokenize_text(query, symbol_to_id, mode="words")
            if not ids:
                return []
            q = torch.tensor([ids], device=device)
            mask = torch.ones_like(q)
            out = self.text_encoder(q, mask)
            return [("phrase:" + query, 1.0, "phrase", out["global_embedding"])]

        # Requête unitaire : mot ou caractère
        from data import _word_key, _char_key, unit_display
        if level == "word":
            key = _word_key(query.strip())
        else:
            key = _char_key(query.strip()) if query.strip() else _char_key(query)

        if key not in symbol_to_id:
            return []

        emb, symbols = self.build_unit_vocab_embeddings(
            {k: v for k, v in symbol_to_id.items()
             if unit_types.get(k, "word" if k.startswith("w:") else "char") == level or level == "all"}
        )
        emb = emb.to(device)

        q_id = torch.tensor([[symbol_to_id[key]]], device=device)
        q_mask = torch.ones_like(q_id)
        q_out = self.text_encoder(q_id, q_mask)
        q_vec = q_out["unit_embeddings"][:, 0]

        sims = q_vec @ emb.t()
        scores, indices = sims.squeeze(0).topk(min(topk, emb.size(0)))
        results = []
        for s, i in zip(scores.tolist(), indices.tolist()):
            sym = symbols[i]
            results.append((unit_display(sym), s, unit_types.get(sym, level)))
        return results

    @torch.no_grad()
    def retrieve_by_query(self, query: str, symbol_to_id: dict,
                          audio_database_embeddings: torch.Tensor,
                          token_mode: str = "words", topk: int = 5):
        """Recherche audio par requête texte (mot, lettre ou phrase)."""
        from data import tokenize_text
        self.eval()
        ids = tokenize_text(query, symbol_to_id, mode=token_mode)
        if not ids:
            return torch.tensor([]), torch.tensor([])
        q = torch.tensor([ids], device=audio_database_embeddings.device)
        mask = torch.ones_like(q)
        return self.retrieve_audio(q, audio_database_embeddings, mask, topk=topk)

    @torch.no_grad()
    def retrieve_audio(self, query_token_ids: torch.Tensor,
                        audio_database_embeddings: torch.Tensor,
                        query_attention_mask: torch.Tensor = None, topk: int = 5):
        """
        query_token_ids : (1, L) la requête texte (ex: "bonjour le monde").
        audio_database_embeddings : (N, D) embeddings globaux précalculés
            pour N segments/fichiers audio indexés.
        Retourne les indices des N meilleurs segments audio correspondants.
        """
        self.eval()
        text_out = self.text_encoder(query_token_ids, query_attention_mask)
        query_vec = text_out["global_embedding"]  # (1, D)
        sims = query_vec @ audio_database_embeddings.t()  # (1, N)
        scores, indices = sims.topk(min(topk, audio_database_embeddings.size(0)), dim=-1)
        return indices.squeeze(0), scores.squeeze(0)

    # ------------------------------------------------------------------ #
    # Utilitaire : construire le lexique vectoriel des unités (lettre isolée)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def build_unit_vocab_embeddings(self, symbol_to_id: dict) -> torch.Tensor:
        """
        Pré-calcule, pour chaque unité du vocabulaire (lettre, phonème, ton),
        son embedding "isolé" dans l'espace latent -- c'est exactement le
        mécanisme décrit en section 5.3 du rapport : le modèle sait
        reconnaître une lettre seule car elle occupe un "quartier" fixe
        de l'espace latent, indépendamment du contexte.
        """
        self.eval()
        symbols = list(symbol_to_id.keys())
        ids = torch.tensor([[symbol_to_id[s]] for s in symbols])  # (V, 1)
        mask = torch.ones_like(ids)
        out = self.text_encoder(ids, mask)
        # Un seul token par séquence -> unit_embeddings[:, 0] == embedding isolé.
        return out["unit_embeddings"][:, 0], symbols  # (V, D), liste des symboles
