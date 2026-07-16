"""
Tête CTC + décodage greedy (pas de décodeur auto-régressif).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    """Projection linéaire frame → vocabulaire CTC (inclut blank=0)."""

    def __init__(self, latent_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(latent_dim, num_classes)

    def forward(self, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        sequence_embeddings : (B, T, D)
        Retourne logits (B, T, C)
        """
        return self.proj(self.dropout(sequence_embeddings))


def build_ctc_char_vocab(symbol_to_id: dict) -> tuple:
    """
    Construit un vocabulaire CTC caractères.
    blank = 0 ; chars = 1..V
    Retourne (ctc_symbol_to_id, id_to_symbol, elus_id_to_ctc)
    """
    chars = sorted(k for k in symbol_to_id if k.startswith("c:"))
    ctc_symbol_to_id = {"<blank>": 0}
    id_to_symbol = {0: "<blank>"}
    elus_id_to_ctc = {}
    idx = 1
    for key in chars:
        ctc_symbol_to_id[key] = idx
        id_to_symbol[idx] = key
        elus_id_to_ctc[symbol_to_id[key]] = idx
        idx += 1
    return ctc_symbol_to_id, id_to_symbol, elus_id_to_ctc


def texts_to_ctc_targets(texts: list, ctc_symbol_to_id: dict) -> tuple:
    """Convertit des transcriptions en listes d'ids CTC (caractères)."""
    targets = []
    lengths = []
    for text in texts:
        ids = []
        for ch in text:
            key = f"c:{ch}"
            if key in ctc_symbol_to_id:
                ids.append(ctc_symbol_to_id[key])
        targets.append(ids)
        lengths.append(len(ids))
    return targets, lengths


def ctc_loss(logits: torch.Tensor, seq_mask: torch.Tensor,
             target_ids: torch.Tensor, target_lens: torch.Tensor) -> torch.Tensor:
    """
    logits     : (B, T, C)
    seq_mask   : (B, T) bool True=valide
    target_ids : (B, L) padded
    target_lens: (B,)
    """
    B, T, C = logits.shape
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # (T, B, C)
    input_lens = seq_mask.long().sum(dim=1).clamp(min=1, max=T)

    # Garder uniquement les exemples où T >= longueur cible (contrainte CTC)
    keep = []
    targets_flat = []
    new_target_lens = []
    new_input_lens = []
    for b in range(B):
        L = int(target_lens[b].item())
        Tin = int(input_lens[b].item())
        if L <= 0 or Tin < L:
            continue
        keep.append(b)
        targets_flat.append(target_ids[b, :L])
        new_target_lens.append(L)
        new_input_lens.append(Tin)

    if not targets_flat:
        return logits.sum() * 0.0

    targets_cat = torch.cat(targets_flat, dim=0)
    # Sous-ensemble des log_probs pour les exemples valides
    idx = torch.tensor(keep, device=logits.device, dtype=torch.long)
    log_probs_k = log_probs.index_select(1, idx)
    in_lens = torch.tensor(new_input_lens, device=logits.device, dtype=torch.long)
    tg_lens = torch.tensor(new_target_lens, device=logits.device, dtype=torch.long)

    return F.ctc_loss(
        log_probs_k, targets_cat, in_lens, tg_lens,
        blank=0, reduction="mean", zero_infinity=True,
    )


@torch.no_grad()
def ctc_greedy_decode(logits: torch.Tensor, seq_mask: torch.Tensor,
                      id_to_symbol: dict) -> list:
    """
    Décodage greedy CTC : argmax → collapse répétitions → drop blank.
    logits : (B, T, C)
    Retourne liste de strings.
    """
    from data import unit_display

    pred = logits.argmax(dim=-1)  # (B, T)
    texts = []
    for b in range(pred.size(0)):
        ids = pred[b][seq_mask[b].bool()].tolist()
        collapsed = []
        prev = None
        for i in ids:
            if i == 0:  # blank
                prev = i
                continue
            if i != prev:
                collapsed.append(i)
            prev = i
        chars = []
        for i in collapsed:
            sym = id_to_symbol.get(i, "")
            if sym and sym != "<blank>":
                chars.append(unit_display(sym))
        texts.append("".join(chars))
    return texts
