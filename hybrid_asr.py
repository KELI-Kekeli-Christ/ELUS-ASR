#!/usr/bin/env python3
"""
ASR hybride Whisper + ELUS (reranking dans l'espace latent).

Compare WER sur test :
  1) Whisper seul (1-best)
  2) Whisper beam top-K reranké par score ELUS (forward pairé audio+texte)
  3) (optionnel) fusion score = alpha*whisper_logprob + beta*elus_sim
"""

import argparse
import json
import os

import torch
import torchaudio
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from data import ParquetAudioTextDataset, collate_fn, load_vocab, tokenize_text, unit_type
from model import ELUSModel
from wer import aggregate_wer, normalize_text, wer


@torch.no_grad()
def load_elus(path: str, device: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ELUSModel(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    sym = ckpt.get("symbol_to_id", {})
    return model, sym


@torch.no_grad()
def elus_paired_score(model, waveform, text, symbol_to_id, device) -> float:
    """Score ELUS : cosinus(global audio, global texte) en forward pairé."""
    ids = tokenize_text(text, symbol_to_id, mode="words")
    if not ids:
        return -1.0
    w = waveform.unsqueeze(0).to(device)
    m = torch.ones_like(w, dtype=torch.long)
    tok = torch.tensor([ids], device=device)
    tmask = torch.ones_like(tok)
    audio_out, text_out = model(w, m, tok, tmask)
    return (audio_out["global_embedding"] * text_out["global_embedding"]).sum().item()


@torch.no_grad()
def elus_candidates(model, waveform, symbol_to_id, device) -> list:
    """Hypothèses ELUS (mots + caractères DTW)."""
    cands = []
    for level in ("word", "char"):
        vocab = {k: v for k, v in symbol_to_id.items() if unit_type(k) == level}
        if not vocab:
            continue
        emb, syms = model.build_unit_vocab_embeddings(vocab)
        emb = emb.to(device)
        w = waveform.unsqueeze(0).to(device)
        hyp, _, _ = model.transcribe_phrase(w, emb, syms, level=level, use_dtw=True)
        if hyp.strip():
            cands.append(hyp.strip())
    return cands


@torch.no_grad()
def whisper_transcribe(model, processor, waveform, sample_rate, device,
                        num_beams: int = 5, n_best: int = 5) -> list:
    """Retourne liste de (texte, score) triée par log-prob Whisper."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    inputs = processor(waveform.squeeze(0).cpu().numpy(), sampling_rate=16000,
                         return_tensors="pt")
    input_features = inputs.input_features.to(device)
    gen_kwargs = {
        "max_new_tokens": 440,
        "num_beams": max(num_beams, n_best),
        "num_return_sequences": min(n_best, max(num_beams, n_best)),
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    out = model.generate(input_features, **gen_kwargs)
    seqs = out.sequences
    results = []
    for i in range(seqs.size(0)):
        text = processor.batch_decode([seqs[i]], skip_special_tokens=True)[0].strip()
        score = 0.0
        if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
            score = out.sequences_scores[i].item()
        results.append((text, score))
    # dédupliquer en gardant meilleur score whisper
    best = {}
    for text, score in results:
        key = normalize_text(text)
        if key and (key not in best or score > best[key][1]):
            best[key] = (text, score)
    ranked = sorted(best.values(), key=lambda x: -x[1])
    return list(ranked)


def pick_hybrid(candidates: list, whisper_scores: dict, elus_scores: dict,
                alpha: float = 0.3, beta: float = 1.0) -> str:
    """Choisit hypothèse avec score fusionné alpha*whisper + beta*elus."""
    best_text, best_score = "", float("-inf")
    for text in candidates:
        key = normalize_text(text)
        if not key:
            continue
        ws = whisper_scores.get(key, -10.0)
        es = elus_scores.get(key, -1.0)
        fused = alpha * ws + beta * es
        if fused > best_score:
            best_score, best_text = fused, text
    return best_text


def load_whisper(model_path: str, processor_path: str, device: str):
    """Charge Whisper fine-tuné (gère dossiers sans preprocessor_config.json)."""
    proc_path = processor_path or model_path
    try:
        processor = WhisperProcessor.from_pretrained(proc_path)
    except OSError:
        from transformers import WhisperFeatureExtractor, WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained(proc_path)
        try:
            feature_extractor = WhisperFeatureExtractor.from_pretrained(proc_path)
        except OSError:
            feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
        processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    return model.to(device).eval(), processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper_model", required=True,
                         help="Chemin modèle Whisper fine-tuné éwé")
    parser.add_argument("--whisper_processor", default=None,
                         help="Chemin processor/tokenizer (si différent du modèle)")
    parser.add_argument("--elus_checkpoint", default="checkpoints/elus_ewe_v3_best.pt")
    parser.add_argument("--elus_vocab", default="checkpoints/symbols.json")
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu",
                         help="Device par défaut (cpu recommandé sur achille)")
    parser.add_argument("--whisper_device", default=None,
                         help="Device Whisper (défaut: --device)")
    parser.add_argument("--elus_device", default=None,
                         help="Device ELUS (défaut: --device, mettre cpu si segfault GPU)")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = tout le test")
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--n_best", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.2,
                         help="Poids score Whisper dans fusion")
    parser.add_argument("--beta", type=float, default=1.0,
                         help="Poids score ELUS dans fusion")
    parser.add_argument("--out", default="logs/hybrid_wer.json")
    args = parser.parse_args()
    whisper_device = args.whisper_device or args.device
    elus_device = args.elus_device or args.device

    print("=" * 60, flush=True)
    print("HYBRID ASR — Whisper vs Whisper+ELUS", flush=True)
    print(f"Whisper : {args.whisper_model} ({whisper_device})", flush=True)
    print(f"ELUS    : {args.elus_checkpoint} ({elus_device})", flush=True)
    print("=" * 60, flush=True)

    elus, sym = load_elus(args.elus_checkpoint, elus_device)
    if not sym:
        sym, _ = load_vocab(args.elus_vocab)

    print("Chargement Whisper...", flush=True)
    proc_path = args.whisper_processor or args.whisper_model
    whisper, processor = load_whisper(args.whisper_model, proc_path, whisper_device)

    ds = ParquetAudioTextDataset(
        args.parquet_dir, sym, split=args.split, token_mode="words",
        max_audio_sec=30.0,
    )
    n = len(ds) if args.max_samples <= 0 else min(args.max_samples, len(ds))
    print(f"Évaluation : {n} utterances ({args.split})", flush=True)

    pairs_whisper = []
    pairs_hybrid = []
    pairs_elus_only = []
    details = []

    for idx in range(n):
        ex = ds[idx]
        ref = ex["text"]
        wav = ex["waveform"]

        w_cands = whisper_transcribe(
            whisper, processor, wav, ds.sample_rate, whisper_device,
            num_beams=args.num_beams, n_best=args.n_best,
        )
        if not w_cands:
            hyp_w = ""
            whisper_scores = {}
        else:
            hyp_w = w_cands[0][0]
            whisper_scores = {normalize_text(t): s for t, s in w_cands}

        elus_cands = elus_candidates(elus, wav, sym, elus_device)
        all_cands = list(dict.fromkeys(
            [hyp_w] + [t for t, _ in w_cands] + elus_cands
        ))

        elus_scores = {}
        for c in all_cands:
            elus_scores[normalize_text(c)] = elus_paired_score(
                elus, wav, c, sym, elus_device,
            )

        hyp_h = pick_hybrid(all_cands, whisper_scores, elus_scores,
                            alpha=args.alpha, beta=args.beta)
        hyp_e = max(all_cands, key=lambda c: elus_scores.get(normalize_text(c), -99))

        pairs_whisper.append((ref, hyp_w))
        pairs_hybrid.append((ref, hyp_h))
        pairs_elus_only.append((ref, hyp_e))

        details.append({
            "idx": idx,
            "ref": ref[:120],
            "whisper": hyp_w[:120],
            "hybrid": hyp_h[:120],
            "wer_w": wer(ref, hyp_w),
            "wer_h": wer(ref, hyp_h),
        })

        if (idx + 1) % 10 == 0 or idx == 0:
            mw = aggregate_wer(pairs_whisper)
            mh = aggregate_wer(pairs_hybrid)
            print(f"  [{idx+1}/{n}] WER whisper={mw['wer']:.1%}  hybrid={mh['wer']:.1%}",
                  flush=True)

    m_w = aggregate_wer(pairs_whisper)
    m_h = aggregate_wer(pairs_hybrid)
    m_e = aggregate_wer(pairs_elus_only)

    print("\n" + "=" * 60, flush=True)
    print(f"RÉSULTATS ({n} utterances, split={args.split})", flush=True)
    print(f"  Whisper seul     : WER={m_w['wer']:.2%}  CER={m_w['cer']:.2%}", flush=True)
    print(f"  Whisper + ELUS   : WER={m_h['wer']:.2%}  CER={m_h['cer']:.2%}", flush=True)
    print(f"  ELUS rerank pur  : WER={m_e['wer']:.2%}  CER={m_e['cer']:.2%}", flush=True)
    delta = m_w["wer"] - m_h["wer"]
    print(f"  Δ WER (positif = hybrid meilleur) : {delta:+.2%}", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    report = {
        "whisper_model": args.whisper_model,
        "elus_checkpoint": args.elus_checkpoint,
        "n": n,
        "split": args.split,
        "whisper": m_w,
        "hybrid": m_h,
        "elus_only": m_e,
        "delta_wer": delta,
        "alpha": args.alpha,
        "beta": args.beta,
        "samples": details[:20],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Rapport : {args.out}", flush=True)


if __name__ == "__main__":
    main()
