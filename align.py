"""
Alignements forcés frame → caractère pour la loss segmentale ELUS.

Génération offline (build_alignments.py) : chemin monotone proportionnel
entre frames audio sous-échantillonnées et indices caractères.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch


def utterance_key(parquet_path: str, row_idx: int) -> str:
    return f"{os.path.basename(parquet_path)}:{row_idx}"


def proportional_char_targets(n_frames: int, n_chars: int) -> List[int]:
    """Assigne à chaque frame l'indice caractère (0..n_chars-1) par mapping monotone."""
    if n_frames <= 0 or n_chars <= 0:
        return []
    return [
        min(n_chars - 1, int(round(i * (n_chars - 1) / max(n_frames - 1, 1))))
        for i in range(n_frames)
    ]


def subsample_targets(targets: List[int], max_frames: int) -> List[int]:
    """Sous-échantillonne les cibles comme dans segmental_infonce_loss."""
    if len(targets) <= max_frames:
        return targets
    idx = torch.linspace(0, len(targets) - 1, max_frames).long().tolist()
    return [targets[i] for i in idx]


def monotonic_dtw_path(sim: torch.Tensor) -> List[int]:
    """
    Chemin DTW monotone sur matrice (T, L) de similarités.
    Retourne pour chaque frame t l'indice colonne j* (0..L-1).
    Transitions : (t,j)->(t+1,j) ou (t,j)->(t+1,j+1).
    """
    if sim.numel() == 0:
        return []
    T, L = sim.shape
    if L == 1:
        return [0] * T

    neg_inf = float("-inf")
    dp = torch.full((T, L), neg_inf, dtype=sim.dtype, device=sim.device)
    bp = torch.zeros(T, L, dtype=torch.long, device=sim.device)

    dp[0, 0] = sim[0, 0]
    for j in range(1, L):
        dp[0, j] = neg_inf

    for t in range(1, T):
        dp[t, 0] = dp[t - 1, 0] + sim[t, 0]
        bp[t, 0] = 0
        for j in range(1, L):
            stay = dp[t - 1, j]
            adv = dp[t - 1, j - 1]
            if adv >= stay:
                dp[t, j] = adv + sim[t, j]
                bp[t, j] = j - 1
            else:
                dp[t, j] = stay + sim[t, j]
                bp[t, j] = j

    j = int(sim.size(1) - 1)
    path = [j] * T
    for t in range(T - 1, 0, -1):
        j = int(bp[t, j].item())
        path[t - 1] = j
    return path


def dtw_decode_units(
    frame_emb: torch.Tensor,
    unit_emb: torch.Tensor,
    min_score: float = -1.0,
) -> Tuple[List[int], List[float]]:
    """
    Décode une séquence d'unités via DTW monotone dans l'espace latent.

    frame_emb : (T, D)
    unit_emb  : (V, D) lexique caractères/mots
    """
    if frame_emb.size(0) == 0 or unit_emb.size(0) == 0:
        return [], []

    sim_tv = frame_emb @ unit_emb.t()
    pred = sim_tv.argmax(1).tolist()

    runs: List[List] = []
    for t, uid in enumerate(pred):
        sc = sim_tv[t, uid].item()
        if sc < min_score:
            continue
        if not runs or runs[-1][0] != uid:
            runs.append([uid, sc, 1])
        else:
            runs[-1][1] += sc
            runs[-1][2] += 1
    if not runs:
        return [], []

    unit_ids = [r[0] for r in runs]
    hyp_emb = unit_emb[unit_ids]
    sim_tl = frame_emb @ hyp_emb.t()
    path = monotonic_dtw_path(sim_tl)

    merged: List[List] = []
    prev_j = None
    for t, j in enumerate(path):
        uid = unit_ids[j]
        sc = sim_tl[t, j].item()
        if sc < min_score:
            continue
        if j != prev_j:
            merged.append([uid, sc, 1])
            prev_j = j
        else:
            merged[-1][1] += sc
            merged[-1][2] += 1

    if not merged:
        return [], []
    return [m[0] for m in merged], [m[1] / m[2] for m in merged]


def dtw_decode_known_units(
    frame_emb: torch.Tensor,
    hyp_unit_emb: torch.Tensor,
    hyp_unit_ids: List[int],
    min_score: float = -1.0,
) -> Tuple[List[int], List[float]]:
    """DTW entre frames et L embeddings hypothèse (ids dans le vocabulaire global)."""
    if frame_emb.size(0) == 0 or hyp_unit_emb.size(0) == 0:
        return [], []

    sim_tl = frame_emb @ hyp_unit_emb.t()
    path = monotonic_dtw_path(sim_tl)

    merged: List[List] = []
    prev_j = None
    for t, j in enumerate(path):
        uid = hyp_unit_ids[j]
        sc = sim_tl[t, j].item()
        if sc < min_score:
            continue
        if j != prev_j:
            merged.append([uid, sc, 1])
            prev_j = j
        else:
            merged[-1][1] += sc
            merged[-1][2] += 1

    if not merged:
        return [], []
    return [m[0] for m in merged], [m[1] / m[2] for m in merged]


class AlignmentStore:
    """Index alignements pré-calculés {clé utterance -> [char_idx,...]}."""

    def __init__(self, path: Optional[str] = None):
        self._data: Dict[str, List[int]] = {}
        if path and os.path.isfile(path):
            self.load(path)

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._data[rec["key"]] = rec["targets"]

    def get(self, key_or_path: str, row_idx: int = 0) -> Optional[List[int]]:
        if isinstance(key_or_path, str) and key_or_path in self._data:
            return self._data[key_or_path]
        return self._data.get(utterance_key(key_or_path, row_idx))

    def __len__(self):
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data
