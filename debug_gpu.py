"""Debug segfault step-by-step on GPU."""
import sys
print("1 start", flush=True)
import torch
print("2 torch", torch.cuda.get_device_name(0), flush=True)
from model import ELUSModel
print("3 model import", flush=True)
from data import ParquetAudioTextDataset, collate_fn
print("4 data import", flush=True)

ckpt = torch.load("checkpoints/elus_ewe_v3_best.pt", map_location="cpu", weights_only=False)
sym = ckpt["symbol_to_id"]
print("5 ckpt", flush=True)

DATA = "/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"
ds = ParquetAudioTextDataset(DATA, sym, split="val", max_files=1)
batch = collate_fn([ds[0]])
print("6 batch", batch["waveforms"].shape, batch["token_ids"].shape, flush=True)

model = ELUSModel(ckpt["config"])
print("7 model init", flush=True)
model.load_state_dict(ckpt["model_state_dict"])
print("8 weights", flush=True)
model = model.cuda().eval()
print("9 cuda", flush=True)

b = {k: v.cuda() for k, v in batch.items()}
print("10 batch cuda", flush=True)
with torch.no_grad():
    a, t = model(b["waveforms"], b["audio_attention_mask"], b["token_ids"], b["text_attention_mask"])
print("11 forward ok", a["global_embedding"].shape, flush=True)
