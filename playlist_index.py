"""
Indexation + recherche de playlist par bribes de paroles (ELUS).

Deux indices :
  - paroles (texte → texte) : idéal si tu te souviens de quelques mots
  - audio   (texte → audio) : utilise l'embedding audio ELUS
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from data import load_vocab, tokenize_text, load_wav_file
from model import ELUSModel

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def load_elus(checkpoint: str, vocab: str, device: str = "cpu"):
    try:
        if torch.cuda.device_count() > 0:
            torch.cuda.init()
    except Exception:
        pass
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ELUSModel(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device).eval()
    symbol_to_id = ckpt.get("symbol_to_id")
    if not symbol_to_id:
        symbol_to_id, _ = load_vocab(vocab)
    return model, symbol_to_id


@torch.no_grad()
def embed_text(model, text: str, symbol_to_id: dict, device: str) -> Optional[torch.Tensor]:
    ids = tokenize_text(text.lower(), symbol_to_id, mode="words")
    if not ids:
        # fallback : caractères
        ids = tokenize_text(text.lower(), symbol_to_id, mode="chars")
    if not ids:
        return None
    q = torch.tensor([ids], device=device)
    mask = torch.ones_like(q)
    out = model.text_encoder(q, mask)
    return out["global_embedding"].squeeze(0).cpu()


def _load_audio(path: str, sample_rate: int, max_audio_sec: float) -> Optional[torch.Tensor]:
    """Charge wav/flac/mp3 (soundfile puis librosa)."""
    try:
        return load_wav_file(path, sample_rate, max_audio_sec)
    except Exception:
        pass
    try:
        import librosa
        audio, _ = librosa.load(path, sr=sample_rate, mono=True, duration=max_audio_sec)
        return torch.from_numpy(audio).float()
    except Exception:
        return None


@torch.no_grad()
def _pool_audio(model, wav: torch.Tensor, device: str) -> torch.Tensor:
    w = wav.unsqueeze(0).to(device)
    m = torch.ones_like(w, dtype=torch.long)
    ao = model.audio_encoder(w, m)
    mask = ao["seq_mask"].unsqueeze(-1)
    pooled = (ao["sequence_embeddings"] * mask).sum(1) / mask.sum(1).clamp(min=1)
    return F.normalize(pooled, dim=-1).squeeze(0).cpu()


@torch.no_grad()
def embed_audio(model, audio_path: str, sample_rate: int, device: str,
                max_audio_sec: float = 30.0) -> Optional[torch.Tensor]:
    wav = _load_audio(audio_path, sample_rate, max_audio_sec)
    if wav is None or wav.numel() < sample_rate // 4:
        return None
    return _pool_audio(model, wav, device)


@torch.no_grad()
def embed_audio_windows(
    model, audio_path: str, sample_rate: int, device: str,
    max_audio_sec: float = 90.0, win_sec: float = 12.0, stride_sec: float = 6.0,
) -> list:
    """Fenêtres audio → embeddings (text→audio sur bribes / milieu de piste)."""
    wav = _load_audio(audio_path, sample_rate, max_audio_sec)
    if wav is None or wav.numel() < sample_rate // 4:
        return []
    win = int(win_sec * sample_rate)
    stride = int(stride_sec * sample_rate)
    if wav.numel() <= win:
        e = _pool_audio(model, wav, device)
        return [e] if e is not None else []
    out = []
    for start in range(0, max(1, wav.numel() - win + 1), stride):
        e = _pool_audio(model, wav[start : start + win], device)
        if e is not None:
            out.append(e)
        if start + win >= wav.numel():
            break
    return out


def find_lyrics(audio_path: Path) -> Optional[str]:
    """Cherche un fichier paroles à côté de l'audio (.txt / .lrc)."""
    for ext in (".txt", ".lrc", ".lyrics"):
        p = audio_path.with_suffix(ext)
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="ignore")
            if ext == ".lrc":
                # strip timestamps [mm:ss.xx]
                import re
                lines = []
                for line in text.splitlines():
                    line = re.sub(r"\[[^\]]*\]", "", line).strip()
                    if line:
                        lines.append(line)
                text = " ".join(lines)
            return text.strip() or None
    return None


def list_audio_files(playlist_dir: str) -> list:
    root = Path(playlist_dir)
    files = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in AUDIO_EXTS and p.is_file():
            files.append(p)
    return files


def lyric_chunks(text: str, win: int = 8, stride: int = 4) -> list:
    """Fenêtres de mots : une bribe de paroles matche mieux un bout qu'un texte entier."""
    words = text.split()
    if not words:
        return []
    if len(words) <= win:
        return [" ".join(words)]
    chunks = []
    for i in range(0, len(words) - win + 1, stride):
        chunks.append(" ".join(words[i : i + win]))
    # toujours garder le texte entier aussi
    chunks.append(" ".join(words))
    # dédoublonner en gardant l'ordre
    seen, out = set(), []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


class PlaylistIndex:
    def __init__(self, model, symbol_to_id, device: str = "cpu"):
        self.model = model
        self.symbol_to_id = symbol_to_id
        self.device = device
        self.sample_rate = getattr(model.cfg, "sample_rate", 16_000)
        self.items = []  # dicts
        self.text_emb = None   # (N, D) optionnel si .txt/.lrc
        self.audio_emb = None  # (N, D) moyenne des fenêtres
        self.chunk_emb = None  # (C, D) fenêtres paroles (optionnel)
        self.chunk_track = None
        self.audio_chunk_emb = None  # (A, D) fenêtres audio ELUS
        self.audio_chunk_track = None

    def build(self, playlist_dir: str, use_whisper: bool = False,
              whisper_model: str = None, max_audio_sec: float = 90.0,
              progress_cb=None):
        """
        Index principal = embeddings audio ELUS (text→audio sur audio non vu).
        Paroles (.txt/.lrc) et Whisper = bonus optionnel, pas requis.
        """
        files = list_audio_files(playlist_dir)
        items = []
        text_vecs = []
        audio_vecs = []
        chunk_vecs = []
        chunk_track = []
        audio_chunk_vecs = []
        audio_chunk_track = []

        whisper = None
        if use_whisper and whisper_model:
            from transformers import pipeline
            whisper = pipeline(
                "automatic-speech-recognition",
                model=whisper_model,
                device=-1,
            )

        dim = self.model.cfg.audio.latent_dim

        for i, path in enumerate(files):
            if progress_cb:
                progress_cb(i / max(len(files), 1), f"Indexation audio : {path.name}")

            title = path.stem.replace("_", " ")
            lyrics = find_lyrics(path)
            if not lyrics and whisper is not None:
                try:
                    out = whisper(str(path))
                    lyrics = (out.get("text") or "").strip()
                except Exception:
                    lyrics = None

            # --- chemin principal ELUS : fenêtres audio ---
            a_wins = embed_audio_windows(
                self.model, str(path), self.sample_rate, self.device,
                max_audio_sec=max_audio_sec,
            )
            if not a_wins:
                a_emb = embed_audio(
                    self.model, str(path), self.sample_rate, self.device, max_audio_sec,
                )
                a_wins = [a_emb] if a_emb is not None else []
            if not a_wins:
                continue

            for e in a_wins:
                audio_chunk_vecs.append(e)
                audio_chunk_track.append(len(items))
            a_emb = F.normalize(torch.stack(a_wins).mean(0), dim=-1)

            # --- bonus : paroles si présentes ---
            t_emb = None
            if lyrics:
                for ch in lyric_chunks(lyrics):
                    e = embed_text(self.model, ch, self.symbol_to_id, self.device)
                    if e is not None:
                        chunk_vecs.append(e)
                        chunk_track.append(len(items))
                t_emb = embed_text(self.model, lyrics, self.symbol_to_id, self.device)

            if t_emb is None:
                t_emb = torch.zeros(dim)

            items.append({
                "path": str(path),
                "title": title,
                "lyrics": lyrics or "",
                "has_lyrics": bool(lyrics),
            })
            text_vecs.append(t_emb)
            audio_vecs.append(a_emb)

        self.items = items
        if items:
            self.text_emb = F.normalize(torch.stack(text_vecs), dim=-1)
            self.audio_emb = F.normalize(torch.stack(audio_vecs), dim=-1)
        else:
            self.text_emb = torch.zeros(0, dim)
            self.audio_emb = torch.zeros(0, dim)

        if chunk_vecs:
            self.chunk_emb = F.normalize(torch.stack(chunk_vecs), dim=-1)
            self.chunk_track = torch.tensor(chunk_track, dtype=torch.long)
        else:
            self.chunk_emb = torch.zeros(0, dim)
            self.chunk_track = torch.zeros(0, dtype=torch.long)

        if audio_chunk_vecs:
            self.audio_chunk_emb = F.normalize(torch.stack(audio_chunk_vecs), dim=-1)
            self.audio_chunk_track = torch.tensor(audio_chunk_track, dtype=torch.long)
        else:
            self.audio_chunk_emb = torch.zeros(0, dim)
            self.audio_chunk_track = torch.zeros(0, dtype=torch.long)

        if progress_cb:
            progress_cb(1.0, f"{len(items)} pistes (audio ELUS)")
        return len(items)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "items": self.items,
            "text_emb": self.text_emb,
            "audio_emb": self.audio_emb,
            "chunk_emb": self.chunk_emb,
            "chunk_track": self.chunk_track,
            "audio_chunk_emb": self.audio_chunk_emb,
            "audio_chunk_track": self.audio_chunk_track,
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.items = data["items"]
        self.text_emb = data["text_emb"]
        self.audio_emb = data["audio_emb"]
        self.chunk_emb = data.get("chunk_emb")
        self.chunk_track = data.get("chunk_track")
        self.audio_chunk_emb = data.get("audio_chunk_emb")
        self.audio_chunk_track = data.get("audio_chunk_track")
        if self.chunk_emb is None:
            self.chunk_emb = torch.zeros(0)
            self.chunk_track = torch.zeros(0, dtype=torch.long)
        if self.audio_chunk_emb is None:
            self.audio_chunk_emb = torch.zeros(0)
            self.audio_chunk_track = torch.zeros(0, dtype=torch.long)
        return len(self.items)

    def _max_over_chunks(self, q, emb, track, n):
        if emb is None or not emb.numel():
            return None
        sims = (q @ emb.t()).squeeze(0)
        scores = torch.full((n,), -1.0)
        for c_i, t_i in enumerate(track.tolist()):
            scores[t_i] = max(scores[t_i].item(), sims[c_i].item())
        return scores

    @torch.no_grad()
    def search(self, query: str, topk: int = 5, mode: str = "audio") -> list:
        """
        mode: audio (défaut ELUS text→audio) | lyrics | hybrid
        """
        if not self.items:
            return []
        q = embed_text(self.model, query, self.symbol_to_id, self.device)
        if q is None:
            return []
        q = F.normalize(q.unsqueeze(0), dim=-1)
        n = len(self.items)

        s_audio = self._max_over_chunks(
            q, self.audio_chunk_emb, self.audio_chunk_track, n,
        )
        if s_audio is None and self.audio_emb is not None and self.audio_emb.numel():
            s_audio = (q @ self.audio_emb.t()).squeeze(0)

        s_text = self._max_over_chunks(q, self.chunk_emb, self.chunk_track, n)
        if s_text is None and self.text_emb is not None and self.text_emb.numel():
            # ignore si tout est zéro (pas de paroles)
            if self.text_emb.abs().sum() > 0:
                s_text = (q @ self.text_emb.t()).squeeze(0)

        if mode == "lyrics" and s_text is not None:
            scores = s_text
        elif mode == "audio" and s_audio is not None:
            scores = s_audio
        else:
            if s_text is not None and s_audio is not None:
                scores = 0.4 * s_text + 0.6 * s_audio
            else:
                scores = s_audio if s_audio is not None else s_text

        k = min(topk, n)
        vals, idxs = scores.topk(k)
        results = []
        for s, i in zip(vals.tolist(), idxs.tolist()):
            it = self.items[i]
            snippet = (it["lyrics"] or it["title"])[:160]
            results.append({
                "rank": len(results) + 1,
                "score": float(s),
                "title": it["title"],
                "path": it["path"],
                "has_lyrics": it["has_lyrics"],
                "snippet": snippet,
            })
        return results
