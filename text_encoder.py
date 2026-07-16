"""
Encodeur Texte E_t
------------------
Transforme une séquence d'unités textuelles (lettres, phonèmes, ou unités
tonales pour l'éwé) en vecteurs situés dans le MÊME espace latent que
l'audio. Fournit à la fois :
  - un vecteur par unité (pour le matching séquentiel / lettre isolée)
  - un vecteur global de la phrase (pour le matching global)
"""

import math
import torch
import torch.nn as nn

from config import TextEncoderConfig


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.embedding_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(cfg.embedding_dim, cfg.max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        self.proj = nn.Sequential(
            nn.LayerNorm(cfg.embedding_dim),
            nn.Linear(cfg.embedding_dim, cfg.latent_dim),
        )

        # Vecteur [CLS] appris pour le résumé global de la phrase.
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.embedding_dim) * 0.02)

    def forward(self, token_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        token_ids: (B, L) indices d'unités texte
        attention_mask: (B, L) 1=valide, 0=padding

        Retourne :
          unit_embeddings  : (B, L, latent_dim)   un vecteur par unité (sans CLS)
          global_embedding : (B, latent_dim)       vecteur [CLS] résumant la phrase
          unit_mask        : (B, L) bool
        """
        B = token_ids.size(0)
        x = self.embed(token_ids)  # (B, L, D)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+L, D)
        x = self.pos_enc(x)

        if attention_mask is not None:
            cls_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=attention_mask.device)
            full_mask = torch.cat([cls_mask, attention_mask], dim=1)
            key_padding_mask = ~full_mask.bool()
        else:
            key_padding_mask = None

        h = self.transformer(x, src_key_padding_mask=key_padding_mask)
        h = self.proj(h)  # (B, 1+L, latent_dim)

        global_emb = torch.nn.functional.normalize(h[:, 0], dim=-1)         # (B, latent_dim)
        unit_emb = torch.nn.functional.normalize(h[:, 1:], dim=-1)          # (B, L, latent_dim)
        unit_mask = attention_mask.bool() if attention_mask is not None else torch.ones(
            token_ids.shape, dtype=torch.bool, device=token_ids.device
        )

        return {
            "unit_embeddings": unit_emb,
            "global_embedding": global_emb,
            "unit_mask": unit_mask,
        }
