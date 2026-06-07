"""v4 loss terms.

Two additions over v3's BCE+Dice(+focal) loss:

1. `tversky_loss` — the imbalance-aware term that v3's config enabled but the
   training loop never actually applied. Tversky Index:
        TI = TP / (TP + alpha*FP + beta*FN)
   with alpha < beta penalizing false negatives harder (recall-favoring), which
   is the standard sparse-positive segmentation recipe (Salehi 2017).

2. `wind_align_loss` — a small regularizer that pushes the predicted new-burn
   centroid downwind of the current fire centroid. This directly attacks the v3
   failure mode where predictions ignored wind and spread radially/urban-ward.

All functions take logits (pre-sigmoid) for numerical stability and operate on
[B,1,H,W] tensors. y is the delta target [B,1,H,W] in {0,1}.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def tversky_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Soft Tversky loss = 1 - TI, averaged over the batch.

    alpha weights false positives, beta weights false negatives.
    alpha=0.3, beta=0.7 => FN penalized ~2.33x more than FP.
    """
    p = torch.sigmoid(logits)
    p = p.flatten(1)          # [B, H*W]
    g = y.flatten(1)
    tp = (p * g).sum(dim=1)
    fp = (p * (1.0 - g)).sum(dim=1)
    fn = ((1.0 - p) * g).sum(dim=1)
    ti = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1.0 - ti).mean()


def _prob_centroid(prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mass-weighted centroid of a [B,1,H,W] probability/mask field.

    Returns [B, 2] as (cx, cy) in pixel coordinates (x = column, y = row).
    Rows increase downward (north is -y in image space); the caller accounts
    for that when comparing to the wind vector.
    """
    b, _, h, w = prob.shape
    ys = torch.arange(h, device=prob.device, dtype=prob.dtype).view(1, 1, h, 1)
    xs = torch.arange(w, device=prob.device, dtype=prob.dtype).view(1, 1, 1, w)
    mass = prob.sum(dim=(1, 2, 3)) + eps                      # [B]
    cx = (prob * xs).sum(dim=(1, 2, 3)) / mass               # [B]
    cy = (prob * ys).sum(dim=(1, 2, 3)) / mass               # [B]
    return torch.stack([cx, cy], dim=-1)                     # [B,2]


def wind_align_loss(
    logits: torch.Tensor,
    fire_last: torch.Tensor,
    u_mean: torch.Tensor,
    v_mean: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize predicted-delta centroid that is not downwind of the fire.

    Parameters
    ----------
    logits : [B,1,H,W]  predicted new-burn logits
    fire_last : [B,1,H,W]  last-frame fire mask (the prior burn footprint)
    u_mean, v_mean : [B]  mean wind components (m/s) over the input sequence,
        in physical space (u = eastward, v = northward).

    Wind blows *toward* (u, v). In image space, +u is +x (east) and +v is -y
    (north is up = decreasing row). So the desired downwind direction in pixel
    space is (u, -v). We penalize 1 - cos(delta_vector, downwind_vector).
    """
    prob = torch.sigmoid(logits)
    fire_centroid = _prob_centroid(fire_last, eps=eps)        # [B,2] (cx,cy)
    pred_centroid = _prob_centroid(prob, eps=eps)             # [B,2]
    delta_vec = pred_centroid - fire_centroid                 # [B,2] (dx,dy)

    # downwind in pixel space: x follows +u, y follows -v (north is up)
    downwind = torch.stack([u_mean, -v_mean], dim=-1)         # [B,2]

    # If there is essentially no wind or no displacement, contribute 0.
    wind_mag = downwind.norm(dim=-1)
    disp_mag = delta_vec.norm(dim=-1)
    active = (wind_mag > 0.5) & (disp_mag > eps)              # >0.5 m/s

    cos = F.cosine_similarity(delta_vec, downwind, dim=-1)    # [B] in [-1,1]
    per_sample = (1.0 - cos)
    per_sample = torch.where(active, per_sample, torch.zeros_like(per_sample))
    denom = active.float().sum().clamp_min(1.0)
    return per_sample.sum() / denom


def v4_total_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    base_loss: torch.Tensor,
    tversky_cfg: Optional[dict] = None,
    wind_cfg: Optional[dict] = None,
    fire_last: Optional[torch.Tensor] = None,
    u_mean: Optional[torch.Tensor] = None,
    v_mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combine the existing BCE+Dice base loss with v4 Tversky + wind-align.

    `base_loss` is the scalar returned by train_nautilus.loss_bce_dice_focal.
    Pass the config sub-dicts straight from CFG.training.loss.
    """
    total = base_loss
    if tversky_cfg and tversky_cfg.get("enable", False):
        total = total + float(tversky_cfg.get("weight", 0.6)) * tversky_loss(
            logits, y,
            alpha=float(tversky_cfg.get("alpha", 0.3)),
            beta=float(tversky_cfg.get("beta", 0.7)),
        )
    if (
        wind_cfg and wind_cfg.get("enable", False)
        and fire_last is not None and u_mean is not None and v_mean is not None
    ):
        total = total + float(wind_cfg.get("weight", 0.1)) * wind_align_loss(
            logits, fire_last, u_mean, v_mean
        )
    return total
