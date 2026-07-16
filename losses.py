"""
Fonctions de perte contrastives
--------------------------------
1. global_infonce_loss      : alignement phrase <-> phrase (paires du batch).
2. segmental_infonce_loss   : alignement segment <-> unité (Temporal Latent
                               Alignment), avec négatifs pris À L'INTÉRIEUR
                               de la même phrase (fenêtre locale) ET dans
                               le reste du batch.

Les deux pertes sont symétriques (Audio->Texte et Texte->Audio), ce qui
matérialise directement la bidirectionnalité décrite dans le rapport.
"""

import torch
import torch.nn.functional as F


def global_infonce_loss(audio_global: torch.Tensor, text_global: torch.Tensor,
                         temperature: float = 0.07) -> torch.Tensor:
    """
    audio_global, text_global : (B, D), déjà normalisés (norme L2 = 1).
    Les paires positives sont les éléments d'indice identique (i, i) dans
    le batch ; toutes les autres paires (i, j != i) servent de négatifs.
    """
    logits = (audio_global.float() @ text_global.float().t()) / temperature  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_a2t = F.cross_entropy(logits, labels)        # audio -> texte
    loss_t2a = F.cross_entropy(logits.t(), labels)    # texte -> audio
    return 0.5 * (loss_a2t + loss_t2a)


def _build_hard_negative_mask(n_segments: int, n_units: int, window: int,
                               device, target: torch.Tensor = None) -> torch.Tensor:
    """Masque booléen (n_segments, n_units) pour négatifs durs locaux."""
    if target is not None:
        seg_idx = torch.arange(n_segments, device=device).float().unsqueeze(1)
        unit_idx = torch.arange(n_units, device=device).float().unsqueeze(0)
        expected_unit = target.float().unsqueeze(1)
        return (unit_idx - expected_unit).abs() <= window

    seg_idx = torch.arange(n_segments, device=device).float().unsqueeze(1)
    unit_idx = torch.arange(n_units, device=device).float().unsqueeze(0)
    expected_unit = seg_idx * (n_units / max(n_segments, 1))
    return (unit_idx - expected_unit).abs() <= window


def _resample_targets(targets: torch.Tensor, n_out: int) -> torch.Tensor:
    """Ré-échantillonne des cibles (M,) vers (n_out,) par interpolation d'indices."""
    if targets.numel() == 0 or n_out <= 0:
        return targets.new_zeros(0, dtype=torch.long)
    if targets.numel() == n_out:
        return targets.long()
    idx = torch.linspace(0, targets.numel() - 1, n_out, device=targets.device).long()
    return targets[idx].long()


def segmental_infonce_loss(audio_seq: torch.Tensor, text_units: torch.Tensor,
                            audio_mask: torch.Tensor, text_mask: torch.Tensor,
                            temperature: float = 0.07,
                            hard_negative_window: int = 3,
                            max_frames: int = 48,
                            seg_targets: torch.Tensor = None) -> torch.Tensor:
    """
    Implémente le "Matching Séquentiel" (section 5 du rapport) :
    chaque segment audio s_i doit être proche du vecteur de l'unité
    texte t_j qu'il prononce, et loin des autres unités -- y compris
    celles de la MÊME phrase (ce qui force une discrimination fine,
    utile pour reconnaître une lettre/un ton isolé).

    audio_seq  : (B, T, D)
    text_units : (B, L, D)
    audio_mask : (B, T) bool, True = segment valide
    text_mask  : (B, L) bool, True = unité valide
    """
    B, T, D = audio_seq.shape
    L = text_units.size(1)
    device = audio_seq.device

    total_loss = audio_seq.new_zeros(())
    n_valid_examples = 0

    for b in range(B):
        s = audio_seq[b][audio_mask[b]]   # (T_b, D)
        t = text_units[b][text_mask[b]]   # (L_b, D)
        if s.size(0) == 0 or t.size(0) == 0:
            continue

        if s.size(0) > max_frames:
            idx = torch.linspace(0, s.size(0) - 1, max_frames, device=device).long()
            s = s[idx]

        if seg_targets is not None:
            st = seg_targets[b]
            st = st[st >= 0]
            if st.numel() > 0:
                target = _resample_targets(st, s.size(0)).clamp(max=t.size(0) - 1)
            else:
                target = torch.round(
                    torch.arange(s.size(0), device=device).float() * (t.size(0) / s.size(0))
                ).long().clamp(max=t.size(0) - 1)
        else:
            target = torch.round(
                torch.arange(s.size(0), device=device).float() * (t.size(0) / s.size(0))
            ).long().clamp(max=t.size(0) - 1)

        sim_local = (s.float() @ t.float().t()) / temperature

        hard_mask = _build_hard_negative_mask(
            s.size(0), t.size(0), hard_negative_window, device, target=target,
        )
        sim_local = sim_local.masked_fill(~hard_mask, float("-inf"))
        sim_local.scatter_(1, target.unsqueeze(1),
                            (s.float() @ t.float().t() / temperature).gather(1, target.unsqueeze(1)))

        loss_local_a2t = F.cross_entropy(sim_local, target)

        total_loss = total_loss + loss_local_a2t
        n_valid_examples += 1

    if n_valid_examples == 0:
        return total_loss

    return total_loss / n_valid_examples


def elus_loss(audio_out: dict, text_out: dict, cfg, text_out_seg: dict = None,
              use_raw_audio: bool = False, seg_targets: torch.Tensor = None) -> dict:
    """Combine perte globale + perte segmentale selon les poids de cfg.LossConfig.
    text_out_seg : sortie encodeur texte pour la loss segmentale (ex. caractères).
    use_raw_audio : si True, segmental sur audio avant matcher (mieux pour chars)."""
    l_global = global_infonce_loss(
        audio_out["global_embedding"], text_out["global_embedding"],
        temperature=cfg.temperature,
    )
    seg_text = text_out_seg if text_out_seg is not None else text_out
    audio_seq = audio_out["sequence_raw"] if use_raw_audio else audio_out["sequence_embeddings"]
    l_seg = segmental_infonce_loss(
        audio_seq, seg_text["unit_embeddings"],
        audio_out["seq_mask"], seg_text["unit_mask"],
        temperature=cfg.temperature,
        hard_negative_window=cfg.hard_negative_window,
        max_frames=cfg.max_segmental_frames,
        seg_targets=seg_targets,
    )
    total = cfg.global_weight * l_global + cfg.segmental_weight * l_seg
    return {"loss": total, "loss_global": l_global.detach(), "loss_segmental": l_seg.detach()}
