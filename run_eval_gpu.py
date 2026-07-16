"""Eval GPU minimal — val/test sur elus_ewe_v3_best.pt"""

import torch
from torch.utils.data import DataLoader

from model import ELUSModel
from data import ParquetAudioTextDataset, collate_fn
from eval import evaluate_loader

DATA = "/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"
CKPT = "/home/data/projets-aps/projet6/new_model/checkpoints/elus_ewe_v3_best.pt"
EVAL_BATCHES = 50
DEVICE = "cuda"


def main():
    print("Device:", DEVICE, torch.cuda.get_device_name(0))
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sym = ckpt["symbol_to_id"]
    print("Saved val metrics:", ckpt.get("val_metrics"), flush=True)

    model = ELUSModel(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()

    for split in ("val", "test"):
        ds = ParquetAudioTextDataset(DATA, sym, split=split, token_mode="words")
        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
        m = evaluate_loader(model, loader, DEVICE, model.cfg, EVAL_BATCHES)
        print(
            "[%s] loss_global=%.4f recall@1=%.3f recall@5=%.3f (%d batches, n=%d)"
            % (split.upper(), m["loss_global"], m["recall@1"], m["recall@5"], EVAL_BATCHES, len(ds))
        )


if __name__ == "__main__":
    main()
