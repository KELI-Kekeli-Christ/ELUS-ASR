"""Test corpus ASR: shortlist + rerank paired."""

import torch
from model import ELUSModel
from data import ParquetAudioTextDataset, collate_fn, tokenize_text

ckpt = torch.load("checkpoints/elus_ewe_v3_best.pt", map_location="cpu", weights_only=False)
sym = ckpt["symbol_to_id"]
model = ELUSModel(ckpt["config"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

DATA = "/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"
test_ds = ParquetAudioTextDataset(DATA, sym, split="test", max_audio_sec=15.0)
train_ds = ParquetAudioTextDataset(DATA, sym, split="train", max_files=5, max_audio_sec=15.0)

# Corpus = transcriptions train (5 fichiers ~ few hundred phrases)
corpus = list({train_ds[i]["text"] for i in range(len(train_ds))})
print(f"Corpus: {len(corpus)} phrases")

idx = 82
ex = test_ds[idx]
ref = ex["text"]
print(f"REF (test#{idx}): {ref[:80]}...")
print(f"REF in corpus: {ref in corpus}")

# Audio-only query vector
one = collate_fn([ex])
with torch.no_grad():
    ao = model.audio_encoder(one["waveforms"], one["audio_attention_mask"])
    mask = ao["seq_mask"].unsqueeze(-1)
    a_q = torch.nn.functional.normalize(
        (ao["sequence_embeddings"] * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1
    )

# Text globals
text_globals = []
for t in corpus:
    ids = tokenize_text(t, sym, mode="words")
    q = torch.tensor([ids])
    m = torch.ones_like(q)
    out = model.text_encoder(q, m)
    text_globals.append(out["global_embedding"])
text_db = torch.cat(text_globals, dim=0)

sims = a_q @ text_db.t()
k = min(30, len(corpus))
top_sc, top_i = sims.squeeze(0).topk(k)

# Rerank top-k with paired forward
best_i, best_s = -1, -1.0
for j in top_i.tolist():
    t = corpus[j]
    item = {"waveform": ex["waveform"], "token_ids": torch.tensor(tokenize_text(t, sym, "words"), dtype=torch.long)}
    batch = collate_fn([{**ex, "token_ids": item["token_ids"]}])
    with torch.no_grad():
        ao2, to2 = model(batch["waveforms"], batch["audio_attention_mask"], batch["token_ids"], batch["text_attention_mask"])
    s = (ao2["global_embedding"] * to2["global_embedding"]).sum().item()
    if s > best_s:
        best_s, best_i = s, j

print(f"\nShortlist top-1: {corpus[top_i[0].item()][:60]}...")
print(f"Rerank best: {corpus[best_i][:80]}...")
print(f"Score: {best_s:.3f}")
