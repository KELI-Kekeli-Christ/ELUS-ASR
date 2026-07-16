"""
Module de Matching Latent -- le "Pont" f(V_a) -> V_t
------------------------------------------------------
Ce n'est PAS un décodeur temporel : c'est un opérateur de projection
sémantique. Deux implémentations sont proposées :

  - "mlp"       : projection indépendante par segment (rapide, baseline).
  - "attention" : chaque segment audio interroge (cross-attention) tous
                  les vecteurs texte disponibles -- plus proche de
                  l'esprit "recherche de proximité" décrit dans le rapport.
"""

import torch
import torch.nn as nn

from config import MatcherConfig


class MLPMatcher(nn.Module):
    def __init__(self, cfg: MatcherConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

    def forward(self, audio_seq: torch.Tensor, text_units: torch.Tensor = None) -> torch.Tensor:
        # audio_seq: (B, T, D) -> projection segment par segment, indépendante du texte.
        out = self.net(audio_seq)
        return torch.nn.functional.normalize(out, dim=-1)


class AttentionMatcher(nn.Module):
    """Cross-attention : chaque segment audio (query) regarde l'ensemble
    des unités texte disponibles dans le batch (key/value) pour affiner
    sa projection -- utile en inférence pour du "grounding" contextuel,
    et facultatif à l'entraînement (peut aussi fonctionner en mode
    "self-only" avec juste un MLP, voir MLPMatcher)."""

    def __init__(self, cfg: MatcherConfig):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.latent_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(cfg.latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )
        self.norm2 = nn.LayerNorm(cfg.latent_dim)

    def forward(self, audio_seq: torch.Tensor, text_units: torch.Tensor,
                text_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        audio_seq: (B, T, D)          -- queries
        text_units: (B, L, D)         -- keys/values (unités texte de la MÊME phrase,
                                          ou d'un lexique élargi en inférence)
        text_key_padding_mask: (B, L) True = position à ignorer (padding)
        """
        attn_out, _ = self.cross_attn(
            query=audio_seq, key=text_units, value=text_units,
            key_padding_mask=text_key_padding_mask,
        )
        x = self.norm1(audio_seq + attn_out)
        x = self.norm2(x + self.ffn(x))
        return torch.nn.functional.normalize(x, dim=-1)


def build_matcher(cfg: MatcherConfig) -> nn.Module:
    if cfg.kind == "mlp":
        return MLPMatcher(cfg)
    elif cfg.kind == "attention":
        return AttentionMatcher(cfg)
    raise ValueError(f"Type de matcher inconnu : {cfg.kind}")
