#!/usr/bin/env python3
"""
Entraînement tête CTC sur encodeur ELUS (éwé).

Pipeline :
  Audio → AfriHuBERT (ELUS) → Linear(V+1) → CTC loss
  Inférence : argmax → collapse → texte

Charge un checkpoint ELUS existant (embeddings), ajoute une tête CTC,
entraîne (tête seule ou + dernières couches audio).
"""

import argparse
import json
import math
import os

import torch

try:
    if torch.cuda.device_count() > 0:
        torch.cuda.init()
except Exception:
    pass

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import ELUSModel
from data import ParquetAudioTextDataset, collate_fn, load_vocab, unit_display
from ctc_head import (
    build_ctc_char_vocab, texts_to_ctc_targets, ctc_loss, ctc_greedy_decode,
)
from wer import aggregate_wer


def freeze_module(module, freeze: bool = True):
    for p in module.parameters():
        p.requires_grad = not freeze


def setup_trainable(model: ELUSModel, train_mode: str):
    """
    train_mode:
      head     — CTC head seule
      last4    — head + 4 dernières couches AfriHuBERT
      full_audio — head + tout l'encodeur audio
    """
    freeze_module(model.text_encoder, True)
    freeze_module(model.matcher, True)
    freeze_module(model.audio_encoder, True)
    freeze_module(model.ctc_head, False)

    if train_mode == "head":
        return
    if train_mode in ("last4", "full_audio"):
        freeze_module(model.audio_encoder, False)
        if train_mode == "last4":
            # regèle : tout gelé sauf dernières couches + proj
            freeze_module(model.audio_encoder, True)
            n = model.cfg.audio.unfreeze_transformer_layers
            for layer in model.audio_encoder.backbone.encoder.layers[-n:]:
                freeze_module(layer, False)
            if hasattr(model.audio_encoder.backbone.encoder, "layer_norm"):
                freeze_module(model.audio_encoder.backbone.encoder.layer_norm, False)
            freeze_module(model.audio_encoder.proj, False)
            if model.audio_encoder.downsample is not None:
                freeze_module(model.audio_encoder.downsample, False)
        freeze_module(model.ctc_head, False)


def collate_ctc(batch, symbol_to_id, ctc_symbol_to_id):
    out = collate_fn(batch, symbol_to_id, with_chars=False)
    texts = [b["text"] for b in batch]
    targets_list, lengths = texts_to_ctc_targets(texts, ctc_symbol_to_id)
    max_l = max(1, max(lengths) if lengths else 1)
    target_ids = torch.zeros(len(batch), max_l, dtype=torch.long)
    for i, ids in enumerate(targets_list):
        if ids:
            target_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    out["ctc_targets"] = target_ids
    out["ctc_target_lens"] = torch.tensor(lengths, dtype=torch.long)
    out["texts"] = texts
    return out


@torch.no_grad()
def evaluate_ctc(model, loader, device, id_to_symbol, max_batches=50):
    model.eval()
    pairs = []
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        waves = batch["waveforms"].to(device)
        amask = batch["audio_attention_mask"].to(device)
        targets = batch["ctc_targets"].to(device)
        tlens = batch["ctc_target_lens"].to(device)
        logits, seq_mask = model.forward_ctc(waves, amask)
        loss = ctc_loss(logits, seq_mask, targets, tlens)
        total_loss += loss.item()
        hyps = ctc_greedy_decode(logits, seq_mask, id_to_symbol)
        for ref, hyp in zip(batch["texts"], hyps):
            pairs.append((ref, hyp))
        n += 1
    metrics = aggregate_wer(pairs)
    metrics["loss_ctc"] = total_loss / max(n, 1)
    model.train()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elus_checkpoint", required=True)
    parser.add_argument("--vocab", default="checkpoints/symbols.json")
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--train_mode", default="head", choices=["head", "last4", "full_audio"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--eval_batches", type=int, default=40)
    parser.add_argument("--max_audio_sec", type=float, default=20.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_path", default="checkpoints/elus_ewe_ctc.pt")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("ELUS + CTC — entraînement tête projection", flush=True)
    print(f"Checkpoint : {args.elus_checkpoint}", flush=True)
    print(f"Mode       : {args.train_mode}", flush=True)
    print("=" * 60, flush=True)

    symbol_to_id, _ = load_vocab(args.vocab)
    ctc_symbol_to_id, id_to_symbol, _ = build_ctc_char_vocab(symbol_to_id)
    num_classes = len(ctc_symbol_to_id)
    print(f"Vocab CTC : {num_classes} classes (blank + {num_classes - 1} chars)", flush=True)

    ckpt = torch.load(args.elus_checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = ELUSModel(cfg)
    # Attacher la tête avant load si le checkpoint en contient déjà une
    has_ctc = any(k.startswith("ctc_head.") for k in ckpt["model_state_dict"])
    model.attach_ctc_head(num_classes)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Load : missing={len(missing)} unexpected={len(unexpected)} "
          f"(ctc_pretrained={has_ctc})", flush=True)
    setup_trainable(model, args.train_mode)
    model = model.to(args.device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Paramètres entraînables : {n_train/1e6:.2f}M / {n_total/1e6:.2f}M", flush=True)

    train_ds = ParquetAudioTextDataset(
        args.parquet_dir, symbol_to_id, split="train", token_mode="words",
        max_audio_sec=args.max_audio_sec,
    )
    val_ds = ParquetAudioTextDataset(
        args.parquet_dir, symbol_to_id, split="val", token_mode="words",
        max_audio_sec=args.max_audio_sec,
    )
    print(f"Train : {len(train_ds)} | Val : {len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=lambda b: collate_ctc(b, symbol_to_id, ctc_symbol_to_id),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_ctc(b, symbol_to_id, ctc_symbol_to_id),
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    use_cuda = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_wer = 1e9
    best_path = args.save_path.replace(".pt", "_best.pt")
    patience = 0
    global_step = 0
    log_loss = 0.0
    log_count = 0

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        for batch in train_loader:
            waves = batch["waveforms"].to(args.device)
            amask = batch["audio_attention_mask"].to(args.device)
            targets = batch["ctc_targets"].to(args.device)
            tlens = batch["ctc_target_lens"].to(args.device)
            if tlens.sum() == 0:
                continue

            with torch.amp.autocast("cuda", enabled=use_cuda):
                logits, seq_mask = model.forward_ctc(waves, amask)
                loss = ctc_loss(logits, seq_mask, targets, tlens) / args.grad_accum

            if not torch.isfinite(loss):
                print(f"[epoch {epoch}] WARN loss non-finie, skip", flush=True)
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                continue

            scaler.scale(loss).backward()
            accum += 1
            log_loss += loss.item() * args.grad_accum
            log_count += 1

            if accum % args.grad_accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        args.grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                accum = 0

                if global_step % 10 == 0 and log_count > 0:
                    print(f"[epoch {epoch}] step {global_step} | "
                          f"loss_ctc={log_loss/log_count:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
                    log_loss, log_count = 0.0, 0

                if args.max_steps and global_step >= args.max_steps:
                    break

        metrics = evaluate_ctc(model, val_loader, args.device, id_to_symbol, args.eval_batches)
        print(f"[epoch {epoch}] VAL | loss_ctc={metrics['loss_ctc']:.4f} "
              f"WER={metrics['wer']:.2%} CER={metrics['cer']:.2%} n={metrics['n']}",
              flush=True)

        # Démo 3 exemples
        model.eval()
        demo = next(iter(val_loader))
        with torch.no_grad():
            logits, sm = model.forward_ctc(
                demo["waveforms"].to(args.device),
                demo["audio_attention_mask"].to(args.device),
            )
            hyps = ctc_greedy_decode(logits, sm, id_to_symbol)
        for ref, hyp in list(zip(demo["texts"], hyps))[:3]:
            print(f"  REF: {ref[:80]!r}", flush=True)
            print(f"  HYP: {hyp[:80]!r}", flush=True)
        model.train()

        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        payload = {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "symbol_to_id": symbol_to_id,
            "ctc_symbol_to_id": ctc_symbol_to_id,
            "id_to_symbol": {str(k): v for k, v in id_to_symbol.items()},
            "val_metrics": metrics,
            "train_mode": args.train_mode,
        }
        torch.save(payload, args.save_path)

        if metrics["wer"] < best_wer - 1e-4:
            best_wer = metrics["wer"]
            patience = 0
            torch.save(payload, best_path)
            print(f"Meilleur CTC : {best_path} (WER={best_wer:.2%})", flush=True)
        else:
            patience += 1
            print(f"Pas d'amélioration WER ({patience}/{args.early_stop_patience})", flush=True)
            if patience >= args.early_stop_patience:
                print("Early stopping.", flush=True)
                break

        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"Terminé. Best WER={best_wer:.2%} → {best_path}", flush=True)


if __name__ == "__main__":
    main()
