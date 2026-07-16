"""Compare les checkpoints sauvegardés sur val/test + recherche mot/lettre."""

import argparse
import glob
import os

import torch
from torch.utils.data import DataLoader

from config import ELUSConfig
from model import ELUSModel
from data import ParquetAudioTextDataset, collate_fn, load_vocab, unit_display
from eval import evaluate_loader


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ELUSModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    symbol_to_id = ckpt.get("symbol_to_id", {})
    val_metrics = ckpt.get("val_metrics", {})
    return model, symbol_to_id, val_metrics


def test_search(model, symbol_to_id, unit_types, queries, device):
    results = {}
    for q, level in queries:
        hits = model.search_units(q, symbol_to_id, unit_types, level=level, topk=3, device=device)
        top = hits[0] if hits else ("?", 0.0, level)
        results[(q, level)] = top
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", default="checkpoints")
    parser.add_argument("--vocab", default="checkpoints/symbols.json")
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_batches", type=int, default=30)
    args = parser.parse_args()

    _, unit_types = load_vocab(args.vocab)
    paths = sorted(glob.glob(os.path.join(args.checkpoints_dir, "elus*.pt")))
    if not paths:
        print("Aucun checkpoint trouvé.")
        return

    cfg = ELUSConfig()
    test_queries = [
        ("ame", "word"),
        ("nye", "word"),
        ("ɖ", "char"),
        ("ƒ", "char"),
    ]

    for split in ("val", "test"):
        print(f"\n{'='*60}")
        print(f"SPLIT : {split}")
        print(f"{'='*60}")

    for path in paths:
        name = os.path.basename(path)
        print(f"\n--- {name} ---")
        try:
            model, symbol_to_id, saved_val = load_checkpoint(path, args.device)
            if not symbol_to_id:
                symbol_to_id, _ = load_vocab(args.vocab)

            for split in ("val", "test"):
                ds = ParquetAudioTextDataset(
                    args.parquet_dir, symbol_to_id, split=split, token_mode="words",
                )
                loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn)
                metrics = evaluate_loader(model, loader, args.device, model.cfg, args.eval_batches)
                print(f"  {split:5s} | loss_global={metrics['loss_global']:.4f} "
                      f"recall@1={metrics['recall@1']:.3f} recall@5={metrics['recall@5']:.3f}")

            if saved_val:
                print(f"  (métriques sauvées dans ckpt: recall@1={saved_val.get('recall@1', '?'):.3f})")

            print("  Recherche unités :")
            for (q, level), (unit, score, utype) in test_search(
                model, symbol_to_id, unit_types, test_queries, args.device
            ).items():
                print(f"    [{level}] '{q}' -> {unit!r} score={score:.3f}")

        except Exception as e:
            print(f"  ERREUR : {e}")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
