"""
Inférence ELUS — transcription, recherche mot/lettre/phrase.
"""

import argparse
import json

import torch
import torchaudio

from model import ELUSModel
from data import load_vocab, tokenize_text, unit_display, unit_type


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ELUSModel(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    symbol_to_id = ckpt.get("symbol_to_id")
    return model, symbol_to_id


def load_waveform(path: str, sample_rate: int, device: str) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.mean(dim=0, keepdim=True).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--audio", type=str, default=None, help="Audio à transcrire")
    parser.add_argument("--query", type=str, default=None,
                         help="Requête texte : mot, lettre ou phrase à chercher")
    parser.add_argument("--search_level", type=str, default="word",
                         choices=["word", "char", "all"],
                         help="Niveau de recherche dans le lexique")
    parser.add_argument("--mode", type=str, default="phrase",
                         choices=["phrase", "segments", "corpus"],
                         help="phrase=CTC mots/lettres, segments=détail frames, corpus=lexique fermé")
    parser.add_argument("--corpus", type=str, default=None,
                         help="Fichier texte (1 phrase/ligne) pour mode corpus")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    symbol_to_id, unit_types = load_vocab(args.vocab)
    model, ckpt_vocab = load_model(args.checkpoint, args.device)
    if ckpt_vocab:
        symbol_to_id = ckpt_vocab

    unit_emb, symbols = model.build_unit_vocab_embeddings(symbol_to_id)
    unit_emb = unit_emb.to(args.device)

    if args.query:
        print(f"Recherche '{args.query}' (niveau={args.search_level}) :")
        if args.search_level == "all":
            ids = tokenize_text(args.query, symbol_to_id, mode="words")
            q = torch.tensor([ids], device=args.device)
            mask = torch.ones_like(q)
            out = model.text_encoder(q, mask)
            print(f"  Embedding phrase prêt (dim={out['global_embedding'].shape[-1]})")
            print("  Utilisez retrieve_by_query() avec une base audio indexée.")
        else:
            results = model.search_units(
                args.query, symbol_to_id, unit_types,
                level=args.search_level, topk=5, device=args.device,
            )
            for unit, score, utype in results:
                print(f"  [{utype}] {unit!r}  score={score:.3f}")

    if args.audio:
        waveform = load_waveform(args.audio, model.cfg.sample_rate, args.device)
        attention_mask = torch.ones_like(waveform, dtype=torch.long)

        if args.search_level == "word":
            filter_types = {"word"}
        elif args.search_level == "char":
            filter_types = {"char"}
        else:
            filter_types = {"word", "char"}

        filtered = {s: symbol_to_id[s] for s in symbol_to_id
                    if unit_types.get(s, unit_type(s)) in filter_types}
        f_emb, f_syms = model.build_unit_vocab_embeddings(filtered)
        f_emb = f_emb.to(args.device)

        if args.mode == "corpus":
            if not args.corpus:
                parser.error("--corpus requis pour mode corpus")
            with open(args.corpus, "r", encoding="utf-8") as f:
                corpus = [ln.strip() for ln in f if ln.strip()]
            phrase, score = model.transcribe_from_corpus(
                waveform, corpus, symbol_to_id, attention_mask=attention_mask,
            )
            print(f"\nTranscription (corpus {len(corpus)} phrases, score={score:.3f}) :")
            print(f"  {phrase}")
        elif args.search_level == "word" and args.mode == "phrase":
            phrase, words, w_scores = model.transcribe_phrase(
                waveform, f_emb, f_syms, attention_mask=attention_mask, level="word",
            )
            print(f"\nTranscription phrase ({len(words)} mots détectés) :")
            print(f"  {phrase}")
            if words:
                print(f"  scores : {[round(s, 3) for s in w_scores]}")
        elif args.search_level == "char" and args.mode == "phrase":
            phrase, chars, c_scores = model.transcribe_phrase(
                waveform, f_emb, f_syms, attention_mask=attention_mask, level="char",
            )
            print(f"\nTranscription phrase ({len(chars)} unités) :")
            print(f"  {phrase}")
        else:
            decoded, scores = model.transcribe(
                waveform, f_emb, f_syms, attention_mask=attention_mask, topk=3,
            )
            print(f"\nTranscription audio (niveau={args.search_level}, top-3/segment) :")
            for t, (candidates, s) in enumerate(zip(decoded[:20], scores.tolist()[:20])):
                labels = [unit_display(c) for c in candidates]
                print(f"  seg {t:03d} -> {labels}  scores={[round(x, 3) for x in s]}")
            if len(decoded) > 20:
                print(f"  ... ({len(decoded)} segments au total)")


if __name__ == "__main__":
    main()
