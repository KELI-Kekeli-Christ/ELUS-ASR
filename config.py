"""
Configuration du système ELUS (Espace Latent Unifié pour l'ASR).
"""

from dataclasses import dataclass, field


@dataclass
class AudioEncoderConfig:
    pretrained_name: str = "ajesujoba/AfriHuBERT"
    freeze_feature_extractor: bool = True
    # Nombre de dernières couches Transformer HuBERT à fine-tuner.
    unfreeze_transformer_layers: int = 4
    hidden_size: int = 768
    latent_dim: int = 256
    # Réduit ~3000 frames -> ~750, mieux aligné avec les mots.
    downsample_factor: int = 4


@dataclass
class TextEncoderConfig:
    vocab_size: int = 100
    embedding_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_len: int = 512
    latent_dim: int = 256


@dataclass
class MatcherConfig:
    kind: str = "attention"
    hidden_dim: int = 512
    num_heads: int = 4
    dropout: float = 0.1
    latent_dim: int = 256


@dataclass
class LossConfig:
    temperature: float = 0.1
    global_weight: float = 1.0
    # Poids réduit : pseudo-alignement mot/frame instable et fait diverger le global.
    segmental_weight: float = 0.15
    hard_negative_window: int = 3
    max_segmental_frames: int = 64
    segmental_on_chars: bool = False
    use_forced_alignments: bool = False


@dataclass
class ELUSConfig:
    audio: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    sample_rate: int = 16_000
    device: str = "cuda"
