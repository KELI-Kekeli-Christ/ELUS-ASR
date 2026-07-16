"""
Test inférence sur audio INCONNU (split test) :
  1) Retrieval texte -> audio
  2) Audio -> texte (retrieval phrase)
"""

import argparse

import torch

try:
    if torch.cuda.device_count() > 0:
        torch.cuda.init()
except Exception:
    pass

from model import ELUSModel
from data import (
    ParquetAudioTextDataset, LibriSpeechDataset, collate_fn, load_vocab,
    tokenize_text, unit_type,
)
from eval import evaluate_loader
from torch.utils.data import DataLoader


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    has_ctc = any(k.startswith("ctc_head.") for k in ckpt["model_state_dict"])
    ctc_n = 0
    if has_ctc and "ctc_symbol_to_id" in ckpt:
        ctc_n = len(ckpt["ctc_symbol_to_id"])
    model = ELUSModel(cfg, ctc_num_classes=ctc_n)
    if has_ctc and model.ctc_head is None:
        model.attach_ctc_head(ctc_n)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(device).eval()
    return model, ckpt.get("symbol_to_id", {}), ckpt.get("val_metrics", {})


@torch.no_grad()
def forward_paired(model, batch, device):
    """Forward identique à l'entraînement (audio + texte de la même phrase)."""
    batch = {k: v.to(device) for k, v in batch.items()}
    ctx = torch.amp.autocast("cuda", enabled=(device == "cuda"))
    with ctx:
        audio_out, text_out = model(
            batch["waveforms"], batch["audio_attention_mask"],
            batch["token_ids"], batch["text_attention_mask"],
        )
    return audio_out["global_embedding"], text_out["global_embedding"]


@torch.no_grad()
def embed_text(model, text, symbol_to_id, device):
    ids = tokenize_text(text, symbol_to_id, mode="words")
    if not ids:
        return None
    q = torch.tensor([ids], device=device)
    mask = torch.ones_like(q)
    ctx = torch.amp.autocast("cuda", enabled=(device == "cuda"))
    with ctx:
        out = model.text_encoder(q, mask)
    return out["global_embedding"]


@torch.no_grad()
def embed_audio_only(model, batch, device):
    """Audio sans texte — pour requête inconnue."""
    batch = {k: v.to(device) for k, v in batch.items()}
    ctx = torch.amp.autocast("cuda", enabled=(device == "cuda"))
    with ctx:
        audio_out = model.audio_encoder(batch["waveforms"], batch["audio_attention_mask"])
        mask = audio_out["seq_mask"].unsqueeze(-1)
        pooled = (audio_out["sequence_embeddings"] * mask).sum(1) / mask.sum(1).clamp(min=1)
    return torch.nn.functional.normalize(pooled, dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/elus_ewe_v3_best.pt")
    parser.add_argument("--vocab", default="checkpoints/symbols.json")
    parser.add_argument("--parquet_dir", default=None)
    parser.add_argument("--librispeech_dir", default=None)
    parser.add_argument("--split", default="test", help="test (Ewe) ou test-clean (LibriSpeech)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pool_size", type=int, default=32)
    parser.add_argument("--n_demo", type=int, default=8)
    parser.add_argument("--max_audio_sec", type=float, default=15.0)
    args = parser.parse_args()

    if not args.parquet_dir and not args.librispeech_dir:
        parser.error("Spécifiez --parquet_dir ou --librispeech_dir")

    print("=" * 60)
    print("TEST INFERENCE — audio jamais vu (split test)")
    print(f"Checkpoint : {args.checkpoint}")
    print("=" * 60)

    model, symbol_to_id, saved = load_model(args.checkpoint, args.device)
    if not symbol_to_id:
        symbol_to_id, _ = load_vocab(args.vocab)

    if args.librispeech_dir:
        test_ds = LibriSpeechDataset(
            args.librispeech_dir, symbol_to_id, split=args.split,
            token_mode="words", max_audio_sec=args.max_audio_sec,
        )
    else:
        test_ds = ParquetAudioTextDataset(
            args.parquet_dir, symbol_to_id, split=args.split, token_mode="words",
            max_audio_sec=args.max_audio_sec,
        )
    print(f"Exemples test : {len(test_ds)}")

    # Métriques officielles (forward pairé, comme entraînement)
    loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    m = evaluate_loader(model, loader, args.device, model.cfg, max_batches=30)
    print(f"\n[TEST batch] recall@1={m['recall@1']:.3f} recall@5={m['recall@5']:.3f} "
          f"loss_global={m['loss_global']:.4f}")

    pool = min(args.pool_size, len(test_ds))
    pool_idxs = list(range(pool))
    batch = collate_fn([test_ds[i] for i in pool_idxs])

    # Index corpus : embeddings pairés (audio+texte connus pour l'index)
    print(f"\n--- INDEX corpus ({pool} clips test) ---")
    audio_db, text_db = forward_paired(model, batch, args.device)
    texts = [test_ds[i]["text"] for i in pool_idxs]

    # 1) RETRIEVAL texte -> audio
    print(f"\n--- RETRIEVAL texte -> audio ({args.n_demo} requêtes) ---")
    t2a_hits = 0
    for idx in range(min(args.n_demo, pool)):
        ref = texts[idx]
        for label, query in [("3 mots", " ".join(ref.split()[:3])), ("phrase", ref)]:
            q = embed_text(model, query, symbol_to_id, args.device)
            sims = q @ audio_db.t()
            rank = sims.argsort(dim=-1, descending=True).squeeze(0).tolist().index(idx) + 1
            hit = rank == 1
            if label == "phrase":
                t2a_hits += int(hit)
            sc = sims[0, idx].item()
            print(f"  clip#{idx} [{label}] rank={rank}/{pool} {'✓' if hit else '✗'} "
                  f"score={sc:.3f} | {query[:50]!r}")
    print(f"Retrieval phrase top-1 : {t2a_hits}/{min(args.n_demo, pool)}")

    # 2) AUDIO -> TEXTE (audio seul, cherche phrase dans corpus)
    print(f"\n--- AUDIO -> TEXTE (audio seul, {args.n_demo} ex. hors pool) ---")
    ood = [pool + 50, pool + 150, pool + 250, pool + 350,
           pool + 450, pool + 550, pool + 650, pool + 750]
    ood = [i for i in ood if i < len(test_ds)][:args.n_demo]

    a2t_hits = 0
    for idx in ood:
        ref = test_ds[idx]["text"]
        one = collate_fn([test_ds[idx]])
        a_q = embed_audio_only(model, one, args.device)
        sims = a_q @ text_db.t()
        top_sc, top_i = sims.squeeze(0).topk(3)
        hyp = texts[top_i[0].item()]
        # Vérifie si la bonne phrase est dans le pool (généralement non)
        hit = hyp == ref
        a2t_hits += int(hit)
        print(f"\n  [test#{idx}] audio inconnu")
        print(f"    REF vraie    : {ref[:95]!r}")
        print(f"    PLUS PROCHE  : {hyp[:95]!r} (score={top_sc[0].item():.3f})")
        for k in range(1, min(3, len(top_i))):
            print(f"    top{k+1}        : {texts[top_i[k].item()][:60]!r} ({top_sc[k].item():.3f})")

    # 3) AUDIO -> TEXTE avec forward pairé (oracle : montre le plafond du modèle)
    print(f"\n--- AUDIO -> TEXTE oracle (forward pairé, leave-one-out {args.n_demo}) ---")
    oracle_hits = 0
    for j, idx in enumerate(range(min(args.n_demo, pool))):
        one = collate_fn([test_ds[idx]])
        a_emb, _ = forward_paired(model, one, args.device)
        sims = a_emb @ text_db.t()
        best = sims.argmax(dim=-1).item()
        hyp = texts[best]
        hit = best == j
        oracle_hits += int(hit)
        print(f"  clip#{idx} {'✓' if hit else '✗'} -> {hyp[:70]!r}...")
    print(f"Oracle top-1 : {oracle_hits}/{min(args.n_demo, pool)}")

    # 4) TRANSCRIPTION phrase (mots détectés -> phrase complète)
    print(f"\n--- TRANSCRIPTION phrase (mots -> phrase, {args.n_demo} ex. test) ---")
    word_vocab = {k: v for k, v in symbol_to_id.items() if unit_type(k) == "word"}
    w_emb, w_syms = model.build_unit_vocab_embeddings(word_vocab)
    w_emb = w_emb.to(args.device)

    overlaps = []
    for idx in ood[:args.n_demo]:
        ref = test_ds[idx]["text"]
        hyp, hyp_words, _ = model.transcribe_phrase(
            test_ds[idx]["waveform"].unsqueeze(0), w_emb, w_syms,
            level="word", use_dtw=True,
        )
        ref_words = ref.split()
        if ref_words:
            ov = len(set(ref_words) & set(hyp_words)) / len(set(ref_words))
        else:
            ov = 0.0
        overlaps.append(ov)
        print(f"\n  [test#{idx}] overlap mots={ov:.0%}")
        print(f"    REF : {ref[:100]!r}")
        print(f"    HYP : {hyp[:100]!r}")

    if overlaps:
        print(f"\nOverlap moyen mots (DTW) : {sum(overlaps)/len(overlaps):.0%}")

    # 5) TRANSCRIPTION caractères (DTW, espace latent)
    print(f"\n--- TRANSCRIPTION caractères (DTW, {args.n_demo} ex. test) ---")
    char_vocab = {k: v for k, v in symbol_to_id.items() if unit_type(k) == "char"}
    c_emb, c_syms = model.build_unit_vocab_embeddings(char_vocab)
    c_emb = c_emb.to(args.device)

    char_overlaps = []
    for idx in ood[:args.n_demo]:
        ref = test_ds[idx]["text"]
        hyp, _, _ = model.transcribe_phrase(
            test_ds[idx]["waveform"].unsqueeze(0), c_emb, c_syms,
            level="char", use_dtw=True,
        )
        ref_c = ref.replace(" ", "")
        hyp_c = hyp.replace(" ", "")
        if ref_c:
            matches = sum(1 for a, b in zip(ref_c, hyp_c) if a == b)
            ov = matches / max(len(ref_c), len(hyp_c))
        else:
            ov = 0.0
        char_overlaps.append(ov)
        print(f"\n  [test#{idx}] char match={ov:.0%}")
        print(f"    REF : {ref[:100]!r}")
        print(f"    HYP : {hyp[:100]!r}")

    if char_overlaps:
        print(f"\nMatch caractères moyen (DTW) : {sum(char_overlaps)/len(char_overlaps):.0%}")

    print("\nTest terminé.")


if __name__ == "__main__":
    main()
