#!/usr/bin/env python3
"""Évalue WER/CER d'un checkpoint ELUS+CTC sur le split test."""

import argparse
import json
import os

import torch

try:
    if torch.cuda.device_count() > 0:
        torch.cuda.init()
except Exception:
    pass

from torch.utils.data import DataLoader

from model import ELUSModel
from data import ParquetAudioTextDataset, load_vocab
from ctc_head import ctc_greedy_decode, build_ctc_char_vocab
from train_ctc import collate_ctc
from wer import aggregate_wer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", default="checkpoints/symbols.json")
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="logs/ctc_test_wer.json")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    symbol_to_id = ckpt.get("symbol_to_id") or load_vocab(args.vocab)[0]
    ctc_symbol_to_id = ckpt.get("ctc_symbol_to_id")
    id_to_symbol = ckpt.get("id_to_symbol")
    if id_to_symbol:
        id_to_symbol = {int(k): v for k, v in id_to_symbol.items()}
    else:
        ctc_symbol_to_id, id_to_symbol, _ = build_ctc_char_vocab(symbol_to_id)

    model = ELUSModel(ckpt["config"], ctc_num_classes=len(ctc_symbol_to_id))
    # compat : attach si absent dans __init__ state
    if model.ctc_head is None:
        model.attach_ctc_head(len(ctc_symbol_to_id))
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(args.device).eval()

    ds = ParquetAudioTextDataset(
        args.parquet_dir, symbol_to_id, split=args.split, token_mode="words",
        max_audio_sec=20.0,
    )
    n = len(ds) if args.max_samples <= 0 else min(args.max_samples, len(ds))
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_ctc(b, symbol_to_id, ctc_symbol_to_id),
    )

    pairs = []
    details = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            if seen >= n:
                break
            logits, sm = model.forward_ctc(
                batch["waveforms"].to(args.device),
                batch["audio_attention_mask"].to(args.device),
            )
            hyps = ctc_greedy_decode(logits, sm, id_to_symbol)
            for ref, hyp in zip(batch["texts"], hyps):
                if seen >= n:
                    break
                pairs.append((ref, hyp))
                if len(details) < 15:
                    details.append({"ref": ref[:120], "hyp": hyp[:120]})
                seen += 1
            if seen % 20 == 0:
                m = aggregate_wer(pairs)
                print(f"[{seen}/{n}] WER={m['wer']:.2%} CER={m['cer']:.2%}", flush=True)

    m = aggregate_wer(pairs)
    print("=" * 50, flush=True)
    print(f"TEST {args.split} n={m['n']}", flush=True)
    print(f"WER={m['wer']:.2%}  CER={m['cer']:.2%}", flush=True)
    print("=" * 50, flush=True)
    for d in details[:5]:
        print(f"REF: {d['ref']!r}", flush=True)
        print(f"HYP: {d['hyp']!r}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "split": args.split,
                    "metrics": m, "samples": details}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
