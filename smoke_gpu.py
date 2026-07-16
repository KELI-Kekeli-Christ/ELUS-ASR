#!/usr/bin/env python3
"""Smoke test GPU — un forward+backward."""
import os
os.environ.setdefault("HF_HOME", "/home/data/projets-aps/projet6/hf_cache")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
from torch.utils.data import DataLoader

from config import ELUSConfig
from model import ELUSModel
from losses import elus_loss
from data import load_vocab, ParquetAudioTextDataset, collate_fn, build_vocab_from_parquet

DATA = "/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"

sym = build_vocab_from_parquet(DATA, "train", max_files=1)
cfg = ELUSConfig()
cfg.text.vocab_size = max(sym.values()) + 1

ds = ParquetAudioTextDataset(DATA, sym, split="train", max_files=1, token_mode="words")
loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)

model = ELUSModel(cfg).cuda().train()
print("embed", model.text_encoder.embed.num_embeddings, "vocab", cfg.text.vocab_size)

for i, batch in enumerate(loader):
    if i > 2:
        break
    batch = {k: v.cuda() for k, v in batch.items()}
    mx = batch["token_ids"].max().item()
    seq_len = batch["text_attention_mask"].sum(dim=1).max().item()
    print(f"batch {i}: tok_max={mx} max_seq={seq_len}")
    assert mx < cfg.text.vocab_size, f"token {mx} >= vocab {cfg.text.vocab_size}"
    audio_out, text_out = model(
        batch["waveforms"], batch["audio_attention_mask"],
        batch["token_ids"], batch["text_attention_mask"],
    )
    losses = elus_loss(audio_out, text_out, cfg.loss)
    losses["loss"].backward()
    print(f"  loss={losses['loss'].item():.4f} global={losses['loss_global'].item():.4f}")

print("OK")
