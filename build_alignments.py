#!/usr/bin/env python3
"""
Pré-calcule les alignements forcés frame → caractère pour l'entraînement ELUS.

Usage Ewe:
  python build_alignments.py --parquet_dir DATA --split train --out alignments/train.jsonl

Usage LibriSpeech:
  python build_alignments.py --librispeech_dir DATA --split train-clean-100 \\
      --vocab checkpoints/symbols_en.json --out alignments/librispeech_train.jsonl
"""

import argparse
import json
import os

import pyarrow.parquet as pq
import torchaudio

from align import utterance_key, proportional_char_targets, subsample_targets
from data import (
    _resolve_parquet_files, tokenize_text, load_vocab,
    build_vocab_from_parquet, build_vocab_from_librispeech,
    load_audio_from_bytes, iter_librispeech_utterances,
)


def estimate_n_frames(n_samples: int, downsample: int = 4, hop: int = 320) -> int:
    t0 = max(1, n_samples // hop)
    return max(1, t0 // downsample)


def write_alignment(fout, key: str, n_samples: int, unit_ids: list,
                    downsample: int, max_segmental_frames: int, level: str = "chars"):
    n_units = len(unit_ids)
    if n_units == 0:
        return False
    n_frames = estimate_n_frames(n_samples, downsample)
    targets = subsample_targets(
        proportional_char_targets(n_frames, n_units), max_segmental_frames,
    )
    rec = {
        "key": key,
        "n_frames": n_frames,
        "n_units": n_units,
        "level": level,
        "targets": targets,
    }
    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", type=str, default=None)
    parser.add_argument("--librispeech_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--sample_rate", type=int, default=16_000)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--max_segmental_frames", type=int, default=128)
    parser.add_argument("--level", type=str, default="chars", choices=["chars", "words"],
                         help="Alignements frame→caractère ou frame→mot")
    args = parser.parse_args()

    if bool(args.parquet_dir) == bool(args.librispeech_dir):
        parser.error("Spécifiez exactement --parquet_dir ou --librispeech_dir")

    tok_mode = "words" if args.level == "words" else "chars"
    if args.vocab:
        symbol_to_id, _ = load_vocab(args.vocab)
    elif args.parquet_dir:
        symbol_to_id = build_vocab_from_parquet(
            args.parquet_dir, args.split, args.max_files,
            include_words=(args.level == "words"),
            include_chars=(args.level == "chars"),
        )
    else:
        symbol_to_id = build_vocab_from_librispeech(
            args.librispeech_dir, args.split,
            include_words=(args.level == "words"),
            include_chars=(args.level == "chars"),
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_written = 0
    n_skipped = 0
    print(f"Alignements niveau={args.level} (token_mode={tok_mode})", flush=True)

    with open(args.out, "w", encoding="utf-8") as fout:
        if args.parquet_dir:
            for fp in _resolve_parquet_files(args.parquet_dir, args.split, args.max_files):
                table = pq.read_table(fp, columns=["transcription", "audio"])
                texts = table["transcription"].to_pylist()
                audios = table["audio"].to_pylist()
                for row_idx, (text, audio) in enumerate(zip(texts, audios)):
                    unit_ids = tokenize_text(text, symbol_to_id, mode=tok_mode)
                    audio_bytes = audio["bytes"]
                    waveform = load_audio_from_bytes(
                        audio_bytes, args.sample_rate, max_audio_sec=30.0,
                    )
                    if waveform is None:
                        n_skipped += 1
                        continue
                    if write_alignment(
                        fout, utterance_key(fp, row_idx), waveform.shape[-1],
                        unit_ids, args.downsample, args.max_segmental_frames, args.level,
                    ):
                        n_written += 1
        else:
            for audio_path, uid, text in iter_librispeech_utterances(args.librispeech_dir, args.split):
                text = text.lower()
                unit_ids = tokenize_text(text, symbol_to_id, mode=tok_mode)
                try:
                    waveform, sr = torchaudio.load(audio_path)
                    if sr != args.sample_rate:
                        waveform = torchaudio.functional.resample(waveform, sr, args.sample_rate)
                    n_samples = waveform.mean(dim=0).shape[-1]
                except Exception:
                    n_skipped += 1
                    continue
                if write_alignment(
                    fout, uid, n_samples, unit_ids,
                    args.downsample, args.max_segmental_frames, args.level,
                ):
                    n_written += 1

    print(f"Alignements écrits : {n_written} (ignorés: {n_skipped}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
