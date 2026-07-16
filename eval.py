"""Métriques d'évaluation ELUS : Recall@K (alignement global) sur un split."""

import torch
from torch.utils.data import DataLoader

from losses import global_infonce_loss


@torch.no_grad()
def recall_at_k(audio_global: torch.Tensor, text_global: torch.Tensor, k: int = 1) -> float:
    """Recall@K symétrique audio<->texte sur un batch (diagonale = positifs)."""
    sim = audio_global @ text_global.t()
    n = sim.size(0)
    if n == 0:
        return 0.0
    labels = torch.arange(n, device=sim.device)
    _, idx_a2t = sim.topk(min(k, n), dim=1)
    _, idx_t2a = sim.t().topk(min(k, n), dim=1)
    r_a2t = (idx_a2t == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    r_t2a = (idx_t2a == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return 0.5 * (r_a2t + r_t2a)


@torch.no_grad()
def evaluate_loader(model, loader, device, cfg, max_batches: int = 50):
    """Évalue Recall@1/@5 et loss globale moyenne sur un DataLoader."""
    model.eval()
    all_audio, all_text = [], []
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        if n_batches >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        if batch["text_attention_mask"].sum() == 0:
            continue
        ctx = torch.amp.autocast("cuda", enabled=(device == "cuda"))
        with ctx:
            audio_out, text_out = model(
                batch["waveforms"], batch["audio_attention_mask"],
                batch["token_ids"], batch["text_attention_mask"],
            )
            loss = global_infonce_loss(
                audio_out["global_embedding"], text_out["global_embedding"],
                temperature=cfg.loss.temperature,
            )
        total_loss += loss.item()
        all_audio.append(audio_out["global_embedding"].cpu())
        all_text.append(text_out["global_embedding"].cpu())
        n_batches += 1

    if n_batches == 0:
        return {"loss_global": 0.0, "recall@1": 0.0, "recall@5": 0.0}

    audio_global = torch.cat(all_audio, dim=0)
    text_global = torch.cat(all_text, dim=0)
    return {
        "loss_global": total_loss / n_batches,
        "recall@1": recall_at_k(audio_global, text_global, k=1),
        "recall@5": recall_at_k(audio_global, text_global, k=5),
    }
