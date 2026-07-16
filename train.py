"""
Script d'entraînement ELUS (v2 — batch effectif large, mots, eval).
"""

import argparse
import json
import math
import os

import torch

# Sur achille : importer data/torchaudio AVANT d'initialiser CUDA provoque un segfault.
try:
    if torch.cuda.device_count() > 0:
        torch.cuda.init()
except Exception:
    pass

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import ELUSConfig
from model import ELUSModel
from losses import elus_loss
from data import (
    AudioTextDataset, ParquetAudioTextDataset, LibriSpeechDataset, collate_fn,
    build_vocab_from_parquet, build_vocab_from_librispeech,
    save_vocab_meta, load_vocab,
)
from align import AlignmentStore
from eval import evaluate_loader


def save_vocab(symbol_to_id: dict, path: str):
    save_vocab_meta(symbol_to_id, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--parquet_dir", type=str, default=None)
    parser.add_argument("--librispeech_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default=None,
                         help="Split validation (ex. dev-clean pour LibriSpeech)")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument("--vocab_out", type=str, default="checkpoints/symbols.json")
    parser.add_argument("--rebuild_vocab", action="store_true")
    parser.add_argument("--token_mode", type=str, default="words",
                         choices=["words", "chars", "both"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4,
                         help="Batch effectif = batch_size * grad_accum")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--segmental_weight", type=float, default=None,
                         help="Poids loss segmentale (défaut: config, 0.15)")
    parser.add_argument("--segmental_on_chars", action="store_true",
                         help="Loss segmentale sur caractères (recommandé v4)")
    parser.add_argument("--alignments", type=str, default=None,
                         help="Fichier JSONL d'alignements forcés (build_alignments.py)")
    parser.add_argument("--max_segmental_frames", type=int, default=None,
                         help="Max frames pour loss segmentale (défaut: config)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_batches", type=int, default=30)
    parser.add_argument("--early_stop_patience", type=int, default=5,
                         help="Arrêt si recall@1 val ne progresse pas pendant N epochs")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                         help="Amélioration minimale de recall@1 pour réinitialiser la patience")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--matcher", type=str, default="attention", choices=["mlp", "attention"])
    parser.add_argument("--save_path", type=str, default="checkpoints/elus.pt")
    parser.add_argument("--audio_pretrained", type=str, default=None,
                         help="Backbone audio HF (ex. facebook/hubert-base-ls960 pour anglais)")
    args = parser.parse_args()

    if not args.manifest and not args.parquet_dir and not args.librispeech_dir:
        parser.error("Spécifiez --manifest, --parquet_dir ou --librispeech_dir")

    if args.vocab and not args.rebuild_vocab:
        symbol_to_id, unit_types = load_vocab(args.vocab)
    elif args.librispeech_dir:
        print("Construction vocabulaire LibriSpeech (mots + lettres)...", flush=True)
        symbol_to_id = build_vocab_from_librispeech(
            args.librispeech_dir, args.split,
            include_words=True, include_chars=True,
        )
        save_vocab(symbol_to_id, args.vocab_out)
        _, unit_types = load_vocab(args.vocab_out)
        n_words = sum(1 for t in unit_types.values() if t == "word")
        n_chars = sum(1 for t in unit_types.values() if t == "char")
        print(f"Vocabulaire : {len(symbol_to_id)} unités ({n_words} mots, {n_chars} lettres)", flush=True)
    elif args.parquet_dir:
        print("Construction du vocabulaire multi-niveau (mots + lettres)...", flush=True)
        symbol_to_id = build_vocab_from_parquet(
            args.parquet_dir, args.split, args.max_files,
            include_words=True, include_chars=True,
        )
        save_vocab(symbol_to_id, args.vocab_out)
        _, unit_types = load_vocab(args.vocab_out)
        n_words = sum(1 for t in unit_types.values() if t == "word")
        n_chars = sum(1 for t in unit_types.values() if t == "char")
        print(f"Vocabulaire : {len(symbol_to_id)} unités ({n_words} mots, {n_chars} lettres)", flush=True)
    else:
        parser.error("Spécifiez --vocab, --parquet_dir ou --librispeech_dir")

    cfg = ELUSConfig()
    cfg.text.vocab_size = max(symbol_to_id.values()) + 1
    if args.audio_pretrained:
        cfg.audio.pretrained_name = args.audio_pretrained
    cfg.matcher.kind = args.matcher
    cfg.device = args.device
    if args.segmental_weight is not None:
        cfg.loss.segmental_weight = args.segmental_weight
    if args.segmental_on_chars:
        cfg.loss.segmental_on_chars = True
    if args.max_segmental_frames is not None:
        cfg.loss.max_segmental_frames = args.max_segmental_frames

    alignment_store = None
    if args.alignments:
        alignment_store = AlignmentStore(args.alignments)
        cfg.loss.use_forced_alignments = True
        print(f"Alignements forcés : {len(alignment_store)} utterances ({args.alignments})",
              flush=True)

    def make_dataset(split):
        if args.librispeech_dir:
            return LibriSpeechDataset(
                args.librispeech_dir, symbol_to_id,
                split=split, sample_rate=cfg.sample_rate,
                token_mode=args.token_mode,
                alignment_store=alignment_store if split == args.split else None,
            )
        if args.parquet_dir:
            return ParquetAudioTextDataset(
                args.parquet_dir, symbol_to_id,
                sample_rate=cfg.sample_rate, split=split,
                max_files=args.max_files, token_mode=args.token_mode,
                alignment_store=alignment_store if split == args.split else None,
            )
        return AudioTextDataset(args.manifest, symbol_to_id,
                                 sample_rate=cfg.sample_rate, token_mode=args.token_mode)

    train_ds = make_dataset(args.split)
    val_split = args.val_split
    if val_split is None and args.librispeech_dir:
        val_split = "dev-clean"
    print(f"Train : {len(train_ds)} exemples | token_mode={args.token_mode}", flush=True)
    print(f"Batch effectif : {args.batch_size * args.grad_accum}", flush=True)
    print(f"Loss weights   : global={cfg.loss.global_weight}, "
          f"segmental={cfg.loss.segmental_weight}, "
          f"segmental_on_chars={cfg.loss.segmental_on_chars}, grad_clip={args.grad_clip}", flush=True)
    print(f"Early stop     : patience={args.early_stop_patience} epochs "
          f"(min_delta recall@1={args.early_stop_min_delta})", flush=True)

    val_loader = None
    if args.parquet_dir or args.librispeech_dir:
        val_ds = make_dataset(val_split) if val_split else make_dataset("val")
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=lambda b: collate_fn(b, symbol_to_id, with_chars=False),
                                 num_workers=0)
        print(f"Val   : {len(val_ds)} exemples", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, symbol_to_id, with_chars=cfg.loss.segmental_on_chars),
        num_workers=args.num_workers, drop_last=True,
    )

    print(f"Chargement AfriHuBERT (downsample={cfg.audio.downsample_factor}, "
          f"unfreeze={cfg.audio.unfreeze_transformer_layers} layers)...", flush=True)
    model = ELUSModel(cfg).to(args.device)
    print(f"Embedding : {model.text_encoder.embed.num_embeddings} tokens "
          f"(vocab_size={cfg.text.vocab_size})", flush=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    use_cuda = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    model.train()
    global_step = 0
    accum_loss = {"loss": 0.0, "loss_global": 0.0, "loss_segmental": 0.0}
    log_loss = {"loss": 0.0, "loss_global": 0.0, "loss_segmental": 0.0}
    log_count = 0
    accum_steps = 0
    best_recall = -1.0
    best_path = args.save_path.replace(".pt", "_best.pt")
    epochs_without_improvement = 0
    stop_training = False

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if batch["text_attention_mask"].sum() == 0:
                continue

            with torch.amp.autocast("cuda", enabled=use_cuda):
                audio_out, text_out = model(
                    batch["waveforms"], batch["audio_attention_mask"],
                    batch["token_ids"], batch["text_attention_mask"],
                )
                text_out_seg = None
                if cfg.loss.segmental_on_chars and "char_token_ids" in batch:
                    text_out_seg = model.text_encoder(
                        batch["char_token_ids"], batch["char_attention_mask"],
                    )
                losses = elus_loss(
                    audio_out, text_out, cfg.loss,
                    text_out_seg=text_out_seg, use_raw_audio=cfg.loss.segmental_on_chars,
                    seg_targets=batch.get("seg_targets"),
                )
                loss = losses["loss"] / args.grad_accum

            if not torch.isfinite(loss):
                print(f"[epoch {epoch}] step {global_step} | WARN loss non-finie, batch ignoré",
                      flush=True)
                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0
                continue

            scaler.scale(loss).backward()
            accum_steps += 1

            for k in log_loss:
                v = losses[k].item() if torch.is_tensor(losses[k]) else losses[k]
                log_loss[k] += v
            log_count += 1

            for k in accum_loss:
                accum_loss[k] += (losses[k].item() if torch.is_tensor(losses[k]) else losses[k]) / args.grad_accum

            if accum_steps % args.grad_accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                accum_steps = 0

                if global_step % 10 == 0 and log_count > 0:
                    lr = scheduler.get_last_lr()[0]
                    avg = {k: log_loss[k] / log_count for k in log_loss}
                    print(f"[epoch {epoch}] step {global_step} | "
                          f"loss={avg['loss']:.4f} "
                          f"(global={avg['loss_global']:.4f}, "
                          f"segmental={avg['loss_segmental']:.4f}) "
                          f"lr={lr:.2e}",
                          flush=True)
                    log_loss = {"loss": 0.0, "loss_global": 0.0, "loss_segmental": 0.0}
                    log_count = 0
                    accum_loss = {"loss": 0.0, "loss_global": 0.0, "loss_segmental": 0.0}

                if args.max_steps and global_step >= args.max_steps:
                    break

        if val_loader is not None:
            metrics = evaluate_loader(model, val_loader, args.device, cfg, args.eval_batches)
            print(f"[epoch {epoch}] VAL | loss_global={metrics['loss_global']:.4f} "
                  f"recall@1={metrics['recall@1']:.3f} recall@5={metrics['recall@5']:.3f}",
                  flush=True)
            if metrics["recall@1"] > best_recall + args.early_stop_min_delta:
                best_recall = metrics["recall@1"]
                epochs_without_improvement = 0
                os.makedirs(os.path.dirname(best_path) or ".", exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                            "symbol_to_id": symbol_to_id, "val_metrics": metrics},
                           best_path)
                print(f"Meilleur checkpoint : {best_path} (recall@1={best_recall:.3f})",
                      flush=True)
            else:
                epochs_without_improvement += 1
                print(f"Pas d'amélioration recall@1 ({epochs_without_improvement}/"
                      f"{args.early_stop_patience})", flush=True)
                if epochs_without_improvement >= args.early_stop_patience:
                    print(f"Early stopping : recall@1 stagne depuis "
                          f"{args.early_stop_patience} epochs (best={best_recall:.3f}).",
                          flush=True)
                    stop_training = True
            model.train()

        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": cfg,
                    "symbol_to_id": symbol_to_id},
                   args.save_path)
        print(f"Checkpoint sauvegardé : {args.save_path}", flush=True)

        if args.max_steps and global_step >= args.max_steps:
            break
        if stop_training:
            break

    if stop_training:
        print(f"Entraînement arrêté tôt. Utiliser : {best_path}", flush=True)


if __name__ == "__main__":
    main()
