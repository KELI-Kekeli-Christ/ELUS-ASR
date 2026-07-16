"""
Encodeur Audio E_a
------------------
Transforme le signal audio brut en une SÉQUENCE d'embeddings temporels
(un vecteur par segment ~20ms), puis les projette dans l'espace latent
partagé avec le texte.

Deux modes de sortie sont exposés :
  - `sequence_embeddings` : (B, T, latent_dim)  -> pour le matching séquentiel
  - `global_embedding`    : (B, latent_dim)      -> pour le matching global (phrase entière)
    obtenu par pooling attentif sur la séquence.
"""

import torch
import torch.nn as nn
from transformers import HubertModel, Wav2Vec2Model

from config import AudioEncoderConfig


class AttentivePooling(nn.Module):
    """Pooling appris (au lieu d'une simple moyenne) pour obtenir un
    vecteur global à partir d'une séquence, en pondérant les segments
    les plus informatifs (ex : voisés vs silences)."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, D)
        scores = self.score(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (weights * x).sum(dim=1)  # (B, D)


class AudioEncoder(nn.Module):
    def __init__(self, cfg: AudioEncoderConfig):
        super().__init__()
        self.cfg = cfg

        # Backbone auto-supervisé pré-entraîné (AfriHuBERT / HuBERT / Wav2Vec2).
        name = cfg.pretrained_name.lower()
        if "hubert" in name or "afrihubert" in name:
            self.backbone = HubertModel.from_pretrained(cfg.pretrained_name)
        else:
            self.backbone = Wav2Vec2Model.from_pretrained(cfg.pretrained_name)

        if cfg.freeze_feature_extractor:
            self.backbone.feature_extractor._freeze_parameters()

        if cfg.unfreeze_transformer_layers > 0:
            for p in self.backbone.parameters():
                p.requires_grad = False
            n = cfg.unfreeze_transformer_layers
            for layer in self.backbone.encoder.layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone.encoder, "layer_norm"):
                for p in self.backbone.encoder.layer_norm.parameters():
                    p.requires_grad = True

        # Sous-échantillonnage temporel optionnel (Conv1d stride>1) pour
        # rapprocher la fréquence des segments audio de celle des unités texte.
        if cfg.downsample_factor > 1:
            self.downsample = nn.Conv1d(
                cfg.hidden_size, cfg.hidden_size,
                kernel_size=cfg.downsample_factor,
                stride=cfg.downsample_factor,
            )
        else:
            self.downsample = None

        # Projection vers l'espace latent partagé.
        self.proj = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Linear(cfg.hidden_size, cfg.latent_dim),
        )

        self.pool = AttentivePooling(cfg.latent_dim)

    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        waveforms: (B, num_samples) audio brut normalisé
        attention_mask: (B, num_samples) 1=valide, 0=padding (optionnel)

        Retourne un dict:
          sequence_embeddings : (B, T, latent_dim)
          global_embedding    : (B, latent_dim)
          seq_mask            : (B, T) bool, True = segment valide
        """
        out = self.backbone(waveforms, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # (B, T0, hidden_size)

        if self.downsample is not None:
            hidden = self.downsample(hidden.transpose(1, 2)).transpose(1, 2)

        seq_emb = self.proj(hidden)  # (B, T, latent_dim)
        seq_emb = torch.nn.functional.normalize(seq_emb, dim=-1)

        if attention_mask is not None:
            seq_mask = self.backbone._get_feature_vector_attention_mask(
                out.last_hidden_state.shape[1], attention_mask
            )
            if self.downsample is not None:
                factor = self.cfg.downsample_factor
                seq_mask = seq_mask[:, ::factor][:, :hidden.shape[1]]
        else:
            seq_mask = torch.ones(seq_emb.shape[:2], dtype=torch.bool, device=seq_emb.device)

        global_emb = self.pool(seq_emb, seq_mask)
        global_emb = torch.nn.functional.normalize(global_emb, dim=-1)

        return {
            "sequence_embeddings": seq_emb,
            "global_embedding": global_emb,
            "seq_mask": seq_mask,
        }
