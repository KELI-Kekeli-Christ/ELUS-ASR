"""Test rapide du checkpoint elus_ewe_v3_best.pt (val/test + recherche + retrieval)."""

import argparse
import random

import torch
from torch.utils.data import DataLoader

from model import ELUSModel
from data import ParquetAudioTextDataset, collate_fn, load_vocab, tokenize_text, unit_display
from eval import evaluate_loader, recall_at_k


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ELUSModel(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model, ckpt.get("symbol_to_id", {}), ckpt.get("val_metrics", {})


@torch.no_grad()
def retrieval_demo(model, ds, symbol_to_id, device, n=5):
    """Texte -> audio : est-ce que le bon clip est dans le top-1 ?"""
    idxs = random.sample(range(len(ds)), min(n, len(ds)))
    ok = 0
    print(f"\n--- Retrieval texte->audio ({n} exemples test) ---")
    for idx in idxs:
        ex = ds[idx]
        text = ex["text"]
        words = [w for w in text.split() if w.strip()][:3]
        query = " ".join(words) if words else text[:40]
        q_ids = tokenize_text(query, symbol_to_id, mode="words")
        if not q_ids:
            continue

        # Index audio : embeddings globaux de n clips
        batch = [ds[i] for i in idxs]
        coll = collate_fn(batch)
        coll = {k: v.to(device) for k, v in coll.items()}
        audio_out, _ = model(coll["waveforms"], coll["audio_attention_mask"],
                             coll["token_ids"], coll["text_attention_mask"])
        db = audio_out["global_embedding"]

        pos = idxs.index(idx)
        q = torch.tensor([q_ids], device=device)
        mask = torch.ones_like(q)
        text_out = model.text_encoder(q, mask)
        q_vec = text_out["global_embedding"]
        sims = q_vec @ db.t()
        rank = sims.argsort(dim=-1, descending=True).squeeze(0).tolist().index(pos) + 1
        hit = rank == 1
        ok += int(hit)
        print(f"  query={query!r:.60} | rank={rank}/{len(idxs)} {'OK' if hit else 'MISS'}")
        print(f"    ref={text[:80]!r}")
    print(f"  Top-1 retrieval: {ok}/{len(idxs)}")
    return ok, len(idxs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/elus_ewe_v3_best.pt")
    parser.add_argument("--vocab", default="checkpoints/symbols.json")
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--max_audio_sec", type=float, default=15.0,
                         help="Tronque l audio pour eviter OOM/segfault sur longs clips")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device     : {args.device}")
    print(f"Checkpoint : {args.checkpoint}")

    model, symbol_to_id, saved = load_model(args.checkpoint, args.device)
    if not symbol_to_id:
        symbol_to_id, unit_types = load_vocab(args.vocab)
    else:
        _, unit_types = load_vocab(args.vocab)

    if saved:
        print(f"Sauvé dans ckpt: recall@1={saved.get('recall@1', '?')}")

    for split in ("val", "test"):
        ds = ParquetAudioTextDataset(
            args.parquet_dir, symbol_to_id, split=split, token_mode="words",
            max_audio_sec=args.max_audio_sec,
        )
        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
        m = evaluate_loader(model, loader, args.device, model.cfg, args.eval_batches)
        print(f"\n[{split.upper()}] loss_global={m['loss_global']:.4f} "
              f"recall@1={m['recall@1']:.3f} recall@5={m['recall@5']:.3f} "
              f"({args.eval_batches} batches)")

    # Recherche mot / lettre dans l'espace latent
    test_ds = ParquetAudioTextDataset(
        args.parquet_dir, symbol_to_id, split="test", token_mode="words",
        max_audio_sec=args.max_audio_sec,
    )
    sample_text = test_ds[0]["text"]
    sample_words = [w for w in sample_text.split() if w.strip()][:4]
    print(f"\n--- Recherche unités (extrait test[0]) ---")
    print(f"  Phrase ref: {sample_text[:100]!r}")

    for w in sample_words:
        for level in ("word", "char"):
            if level == "word":
                q = w
            else:
                q = w[0] if w else ""
            if not q:
                continue
            hits = model.search_units(q, symbol_to_id, unit_types, level=level, topk=3, device=args.device)
            line = ", ".join(f"{unit_display(u)!r}({s:.3f})" for u, s, _ in hits)
            print(f"  [{level}] '{q}' -> {line}")

    # Retrieval sur sous-ensemble test
    retrieval_demo(model, test_ds, symbol_to_id, args.device, n=8)

    print("\nTest terminé.")


if __name__ == "__main__":
    main()
