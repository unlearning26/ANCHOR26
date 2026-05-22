# SR-CIB (Structural-Redundancy Calibrated, Information-Balanced) loss
# Supervised semantic segmentation (single-model training) with interchangeable redundancy proxies.
#
# Key design choices (aligned with our contract):
# - Training framework: single-model supervised (logits -> softmax -> CE + Dice).
# - SR-CIB acts as a *reweighting of voxel-wise evidence* via redundancy proxy r(v).
# - Proxies A–E implemented and selectable via a flag.
# - Guardrails: warm-up, weight clipping/floor, stop-gradient through weights (default).
# - Works for 2D (B,C,H,W) and 3D (B,C,D,H,W). Video can be treated as 3D (T,H,W).

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def _is_3d(logits: torch.Tensor) -> bool:
    # logits: (B,C,H,W) or (B,C,D,H,W)
    return logits.dim() == 5

def _spatial_dims(logits: torch.Tensor) -> Tuple[int, ...]:
    return tuple(range(2, logits.dim()))

def _avg_pool_nd(x: torch.Tensor, kernel_size: Tuple[int, ...]) -> torch.Tensor:
    """Channel-wise average pooling for 2D/3D tensors.
    x: (B,C,...) -> returns same shape.
    """
    if x.dim() == 4:
        # (B,C,H,W)
        kH, kW = kernel_size
        return F.avg_pool2d(x, kernel_size=(kH, kW), stride=1, padding=(kH // 2, kW // 2))
    elif x.dim() == 5:
        # (B,C,D,H,W)
        kD, kH, kW = kernel_size
        return F.avg_pool3d(
            x,
            kernel_size=(kD, kH, kW),
            stride=1,
            padding=(kD // 2, kH // 2, kW // 2),
        )
    else:
        raise ValueError(f"Expected 4D or 5D tensor, got {x.dim()}D.")

def _normalize_to_unit_interval(x: torch.Tensor, eps: float = 1e-6,
                                mode: Literal["minmax", "percentile"] = "percentile",
                                p_lo: float = 5.0, p_hi: float = 95.0) -> torch.Tensor:
    """Normalize per-sample (per-batch element) to [0,1] for stability.
    x: (B,1,...) or (B,...) -> returns same shape.
    """
    if x.dim() < 2:
        raise ValueError("x must have batch dimension.")

    # Work per-sample to avoid batch coupling.
    B = x.shape[0]
    x_flat = x.view(B, -1)

    if mode == "minmax":
        lo = x_flat.min(dim=1).values.view(B, *([1] * (x.dim() - 1)))
        hi = x_flat.max(dim=1).values.view(B, *([1] * (x.dim() - 1)))
    elif mode == "percentile":
        # Percentiles per sample (approx via kthvalue)
        n = x_flat.shape[1]
        k_lo = max(1, int(round((p_lo / 100.0) * (n - 1))) + 1)
        k_hi = max(1, int(round((p_hi / 100.0) * (n - 1))) + 1)
        lo = x_flat.kthvalue(k_lo, dim=1).values.view(B, *([1] * (x.dim() - 1)))
        hi = x_flat.kthvalue(k_hi, dim=1).values.view(B, *([1] * (x.dim() - 1)))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    denom = (hi - lo).clamp_min(eps)
    return ((x - lo) / denom).clamp(0.0, 1.0)

def _safe_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """p: (B,C,...) probabilities."""
    return -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=1, keepdim=True)

def _make_onehot(target: torch.Tensor, num_classes: int, ignore_index: Optional[int]) -> torch.Tensor:
    """target: (B,...) int64 -> onehot: (B,C,...) float"""
    if target.dtype != torch.long:
        target = target.long()

    if ignore_index is None:
        oh = F.one_hot(target, num_classes=num_classes)  # (B, ..., C)
        oh = oh.permute(0, -1, *range(1, target.dim())).contiguous()
        return oh.float()

    # If ignore_index exists, set ignored positions to 0 in onehot.
    valid = (target != ignore_index)
    safe_target = target.clone()
    safe_target[~valid] = 0
    oh = F.one_hot(safe_target, num_classes=num_classes)  # (B, ..., C)
    oh = oh.permute(0, -1, *range(1, target.dim())).contiguous().float()
    oh = oh * valid.unsqueeze(1).float()
    return oh

def _make_valid_mask(target: torch.Tensor, ignore_index: Optional[int]) -> torch.Tensor:
    """Returns mask: (B,1,...) float in {0,1}."""
    if ignore_index is None:
        return torch.ones((target.shape[0], 1, *target.shape[1:]), device=target.device, dtype=torch.float32)
    return (target != ignore_index).unsqueeze(1).float()


# -----------------------------
# Redundancy proxy implementations (A–E)
# -----------------------------

@dataclass
class RedundancyConfig:
    proxy: Literal["A_prob_coherence", "B_local_moran", "C_geary", "D_structure_tensor", "E_grad_coherence"] = "A_prob_coherence"
    # Neighborhood kernel sizes (2D: (kH,kW), 3D: (kD,kH,kW))
    kernel_size_2d: Tuple[int, int] = (7, 7)
    kernel_size_3d: Tuple[int, int, int] = (3, 7, 7)  # anisotropic default (thicker in-plane)
    # Scalar field choice for Moran/Geary/StructureTensor
    scalar_field: Literal["max_prob", "fg_prob", "entropy"] = "max_prob"
    fg_class: int = 1  # used if scalar_field="fg_prob"
    # Normalization for redundancy map r
    norm_mode: Literal["minmax", "percentile"] = "percentile"
    p_lo: float = 5.0
    p_hi: float = 95.0
    eps: float = 1e-6

class RedundancyComputer(nn.Module):
    def __init__(self, cfg: RedundancyConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        logits: torch.Tensor,          # (B,C,...) raw logits
        probs: torch.Tensor,           # (B,C,...) softmax
        target: Optional[torch.Tensor] # (B,...) labels (needed for E; optional for others)
    ) -> torch.Tensor:
        """Return redundancy r in [0,1], shape: (B,1,...)"""
        if logits.dim() not in (4, 5):
            raise ValueError(f"logits must be 4D/5D, got {logits.shape}")

        k = self.cfg.kernel_size_3d if _is_3d(logits) else self.cfg.kernel_size_2d

        if self.cfg.proxy == "A_prob_coherence":
            # r(v) = <P(v), K*P(v)> summed over classes (inner product with neighbor-averaged probs)
            p_smooth = _avg_pool_nd(probs, k)
            r = (probs * p_smooth).sum(dim=1, keepdim=True)  # (B,1,...)
            r = _normalize_to_unit_interval(r, eps=self.cfg.eps, mode=self.cfg.norm_mode, p_lo=self.cfg.p_lo, p_hi=self.cfg.p_hi)
            return r

        if self.cfg.proxy in ("B_local_moran", "C_geary", "D_structure_tensor"):
            p_scalar = self._scalar_from_probs(probs)  # (B,1,...)
            if self.cfg.proxy == "B_local_moran":
                return self._local_moran(p_scalar, k)
            if self.cfg.proxy == "C_geary":
                return self._geary(p_scalar, k)
            # D_structure_tensor
            return self._structure_tensor_coherence(p_scalar, k)

        if self.cfg.proxy == "E_grad_coherence":
            if target is None:
                raise ValueError("Proxy E requires target labels (supervised).")
            return self._grad_coherence(probs, target, k)

        raise ValueError(f"Unknown proxy: {self.cfg.proxy}")

    def _scalar_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        # probs: (B,C,...)
        if self.cfg.scalar_field == "max_prob":
            return probs.max(dim=1, keepdim=True).values
        if self.cfg.scalar_field == "entropy":
            # higher entropy -> less coherence; for redundancy we’ll invert later via normalization + mapping
            return _safe_entropy(probs, eps=self.cfg.eps)
        if self.cfg.scalar_field == "fg_prob":
            c = int(self.cfg.fg_class)
            if c < 0 or c >= probs.shape[1]:
                raise ValueError(f"fg_class {c} out of range for C={probs.shape[1]}")
            return probs[:, c:c+1, ...]
        raise ValueError(f"Unknown scalar_field: {self.cfg.scalar_field}")

    def _local_moran(self, x: torch.Tensor, k: Tuple[int, ...]) -> torch.Tensor:
        """Local Moran's I on scalar field x (B,1,...). Returns redundancy r in [0,1] where higher means more clustered."""
        # Global mean/var per-sample
        B = x.shape[0]
        x_flat = x.view(B, -1)
        mu = x_flat.mean(dim=1).view(B, *([1] * (x.dim() - 1)))
        xc = x - mu
        s2 = (xc.view(B, -1).pow(2).mean(dim=1).view(B, *([1] * (x.dim() - 1)))).clamp_min(self.cfg.eps)

        # Neighbor mean of centered values
        nb = _avg_pool_nd(xc, k)

        I = (xc / s2) * nb  # (B,1,...), can be negative; clustered positive -> positive
        # Map to [0,1] redundancy: larger positive I => more redundancy
        I = _normalize_to_unit_interval(I, eps=self.cfg.eps, mode=self.cfg.norm_mode, p_lo=self.cfg.p_lo, p_hi=self.cfg.p_hi)
        return I

    def _geary(self, x: torch.Tensor, k: Tuple[int, ...]) -> torch.Tensor:
        """Local Geary's C-like dissimilarity: average squared difference to neighbor mean.
        Returns redundancy r in [0,1] where higher means more redundant (more similar).
        """
        nb = _avg_pool_nd(x, k)
        dis = (x - nb).pow(2)  # high dissimilarity
        dis = _normalize_to_unit_interval(dis, eps=self.cfg.eps, mode=self.cfg.norm_mode, p_lo=self.cfg.p_lo, p_hi=self.cfg.p_hi)
        r = 1.0 - dis  # high similarity => high redundancy
        return r.clamp(0.0, 1.0)

    def _structure_tensor_coherence(self, x: torch.Tensor, k: Tuple[int, ...]) -> torch.Tensor:
        """Structure-tensor coherence proxy (lightweight).
        We approximate coherence using gradient magnitudes and anisotropy of local gradients.
        For efficiency, we use finite differences and local averaging.
        Returns redundancy in [0,1] where higher means more coherent/structured neighborhood.
        """
        # Finite differences (forward diff) with padding
        if x.dim() == 4:
            # (B,1,H,W)
            dx = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
            dy = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
            # Local second-moment matrix entries (smoothed)
            Jxx = _avg_pool_nd(dx * dx, k)
            Jyy = _avg_pool_nd(dy * dy, k)
            Jxy = _avg_pool_nd(dx * dy, k)
            # Coherence: (λ1-λ2)/(λ1+λ2)
            trace = (Jxx + Jyy).clamp_min(self.cfg.eps)
            det = (Jxx * Jyy - Jxy * Jxy).clamp_min(0.0)
            # eigenvalues of 2x2: λ = (tr ± sqrt(tr^2 - 4 det))/2
            disc = (trace * trace - 4.0 * det).clamp_min(0.0).sqrt()
            l1 = 0.5 * (trace + disc)
            l2 = 0.5 * (trace - disc)
            coh = ((l1 - l2) / (l1 + l2 + self.cfg.eps)).clamp(0.0, 1.0)
            coh = _normalize_to_unit_interval(coh, eps=self.cfg.eps, mode=self.cfg.norm_mode, p_lo=self.cfg.p_lo, p_hi=self.cfg.p_hi)
            return coh

        # 3D (B,1,D,H,W)
        dz = F.pad(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))
        dy = F.pad(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], (0, 0, 0, 1, 0, 0))
        dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))

        # second moments (smoothed)
        Jxx = _avg_pool_nd(dx * dx, k)
        Jyy = _avg_pool_nd(dy * dy, k)
        Jzz = _avg_pool_nd(dz * dz, k)
        # For efficiency, approximate coherence by anisotropy ratio:
        # coherence ~ (max - mean)/(max + eps)
        stack = torch.cat([Jxx, Jyy, Jzz], dim=1)  # (B,3,...)
        mx = stack.max(dim=1, keepdim=True).values
        mn = stack.mean(dim=1, keepdim=True)
        coh = ((mx - mn) / (mx + self.cfg.eps)).clamp(0.0, 1.0)
        coh = _normalize_to_unit_interval(coh, eps=self.cfg.eps, mode=self.cfg.norm_mode, p_lo=self.cfg.p_lo, p_hi=self.cfg.p_hi)
        return coh

    def _grad_coherence(self, probs: torch.Tensor, target: torch.Tensor, k: Tuple[int, ...]) -> torch.Tensor:
        """Proxy E: coherence of per-voxel CE gradient w.r.t logits.
        For CE with softmax: ∂CE/∂z = p - y_onehot (ignoring class weights).
        We compute local agreement of normalized gradient vectors.
        """
        B, C = probs.shape[:2]
        onehot = _make_onehot(target, num_classes=C, ignore_index=None).to(probs.dtype).to(probs.device)  # (B,C,...)
        g = probs - onehot  # (B,C,...)
        # Normalize gradient vector per voxel to unit norm (cosine similarity proxy)
        g_norm = g / (g.square().sum(dim=1, keepdim=True).clamp_min(self.cfg.eps).sqrt())
        g_smooth = _avg_pool_nd(g_norm, k)
        r = (g_norm * g_smooth).sum(dim=1, keepdim=True)  # cosine with neighbor-average
        r = _normalize_to_unit_interval(r, eps=self.cfg.eps, mode=self.cfg.norm_mode, p_lo=self.cfg.p_lo, p_hi=self.cfg.p_hi)
        return r


# -----------------------------
# SR-CIB Loss (CE + Dice)
# -----------------------------

@dataclass
class SRCIBConfig:
    # Redundancy proxy config
    red: RedundancyConfig = RedundancyConfig()

    # Loss weights
    lambda_ce: float = 1.0
    lambda_dice: float = 1.0

    # SR-CIB weighting parameters
    delta_max: float = 2.0              # strength exponent
    warmup_steps: int = 2000            # ramp from 0 to delta_max over this many steps
    w_min: float = 0.3                  # lower bound on weights (guardrail)
    detach_weights: bool = True         # stop-grad through weights (strongly recommended)
    # weight mapping: w = clip((1 - r_norm)^delta, w_min, 1)
    # where r_norm in [0,1]

    # CE details
    ignore_index: Optional[int] = None
    # Optional class weights for CE
    ce_class_weights: Optional[torch.Tensor] = None  # shape (C,)

    # Dice details
    dice_smooth: float = 1e-5
    dice_include_background: bool = True
    dice_weighted: bool = False         # if True, weight Dice by the same w(v) (advanced; default False)

class SRCIBLoss(nn.Module):
    """Supervised SR-CIB loss for semantic segmentation.
    - logits: (B,C,H,W) or (B,C,D,H,W)
    - target: (B,H,W) or (B,D,H,W) with class indices
    Returns: (loss, stats_dict)
    """
    def __init__(self, cfg: SRCIBConfig):
        super().__init__()
        self.cfg = cfg
        self.red_comp = RedundancyComputer(cfg.red)

    def _delta(self, global_step: int) -> float:
        if self.cfg.warmup_steps <= 0:
            return float(self.cfg.delta_max)
        t = min(max(global_step, 0), self.cfg.warmup_steps)
        return float(self.cfg.delta_max) * float(t) / float(self.cfg.warmup_steps)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        global_step: int = 0,
        return_maps: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if logits.dim() not in (4, 5):
            raise ValueError(f"logits must be 4D or 5D, got {logits.shape}")
        if target.dim() != logits.dim() - 1:
            raise ValueError(f"target must have shape (B,spatial...), got {target.shape} vs logits {logits.shape}")

        B, C = logits.shape[:2]
        probs = F.softmax(logits, dim=1)

        valid_mask = _make_valid_mask(target, self.cfg.ignore_index)  # (B,1,...)

        # Compute redundancy r in [0,1], shape (B,1,...)
        r = self.red_comp(logits=logits, probs=probs, target=target if self.cfg.red.proxy == "E_grad_coherence" else None)

        # Normalize and weight
        # If scalar_field was entropy, high entropy means low coherence; our normalization maps it anyway.
        delta = self._delta(global_step)
        w = (1.0 - r).clamp(0.0, 1.0).pow(delta)  # (B,1,...)
        w = w.clamp(min=self.cfg.w_min, max=1.0)
        w = w * valid_mask  # zero out ignored voxels
        if self.cfg.detach_weights:
            w = w.detach()

        # --- Weighted CE (voxelwise) ---
        ce = self._weighted_ce(logits, target, w)  # scalar

        # --- Dice ---
        dice = self._dice_loss(probs, target, w if self.cfg.dice_weighted else None, valid_mask)

        loss = self.cfg.lambda_ce * ce + self.cfg.lambda_dice * dice

        stats = {
            "loss_total": float(loss.detach().cpu()),
            "loss_ce": float(ce.detach().cpu()),
            "loss_dice": float(dice.detach().cpu()),
            "delta": float(delta),
            "w_mean": float((w.sum() / (valid_mask.sum().clamp_min(1.0))).detach().cpu()),
            "r_mean": float((r.sum() / (valid_mask.sum().clamp_min(1.0))).detach().cpu()),
            "w_min_actual": float(w.min().detach().cpu()),
            "w_max_actual": float(w.max().detach().cpu()),
            "proxy": self.cfg.red.proxy,
        }

        if return_maps:
            stats["r_map"] = r
            stats["w_map"] = w

        return loss, stats

    def _weighted_ce(self, logits: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # per-voxel CE
        # F.cross_entropy supports ignore_index and class weights, but returns reduced scalar.
        # We need unreduced per-voxel loss -> use log-softmax + gather.
        logp = F.log_softmax(logits, dim=1)  # (B,C,...)

        # target gather: expand to (B,1,...)
        t = target.clone()
        if self.cfg.ignore_index is not None:
            # replace ignored targets with 0 for gather, will be masked by w anyway
            t[t == self.cfg.ignore_index] = 0
        t = t.unsqueeze(1)  # (B,1,...)

        nll = -logp.gather(dim=1, index=t).squeeze(1)  # (B,...)

        # class weights (optional)
        if self.cfg.ce_class_weights is not None:
            cw = self.cfg.ce_class_weights.to(logits.device, dtype=logits.dtype)  # (C,)
            # weight per voxel by class
            vw = cw.gather(dim=0, index=target.clamp(min=0, max=C-1).view(-1)).view_as(nll)
            if self.cfg.ignore_index is not None:
                vw = vw * (target != self.cfg.ignore_index).to(vw.dtype)
            nll = nll * vw

        # apply SR-CIB weights
        # w: (B,1,...) -> squeeze channel
        w_s = w.squeeze(1)
        num = (w_s * nll).sum()
        den = w_s.sum().clamp_min(1.0)
        return num / den

    def _dice_loss(
        self,
        probs: torch.Tensor,             # (B,C,...)
        target: torch.Tensor,            # (B,...)
        w: Optional[torch.Tensor],       # (B,1,...) or None
        valid_mask: torch.Tensor,        # (B,1,...)
    ) -> torch.Tensor:
        B, C = probs.shape[:2]
        onehot = _make_onehot(target, num_classes=C, ignore_index=self.cfg.ignore_index).to(probs.dtype).to(probs.device)  # (B,C,...)

        if not self.cfg.dice_include_background and C > 1:
            probs = probs[:, 1:, ...]
            onehot = onehot[:, 1:, ...]
            C = C - 1

        # valid mask broadcast
        vm = valid_mask
        if not self.cfg.dice_include_background and valid_mask is not None:
            vm = valid_mask  # still applies to all voxels; ok

        if w is None:
            # unweighted dice (but masked)
            probs_m = probs * vm
            onehot_m = onehot * vm
            inter = (probs_m * onehot_m).sum(dim=_spatial_dims(probs_m))
            den = (probs_m + onehot_m).sum(dim=_spatial_dims(probs_m))
        else:
            # weighted dice: weights broadcast to classes
            w_c = w.expand(-1, probs.shape[1], *([-1] * (probs.dim() - 2))) if w.dim() == probs.dim() else w
            probs_m = probs * vm * w_c
            onehot_m = onehot * vm * w_c
            inter = (probs_m * onehot_m).sum(dim=_spatial_dims(probs_m))
            den = (probs_m + onehot_m).sum(dim=_spatial_dims(probs_m))

        dice = (2.0 * inter + self.cfg.dice_smooth) / (den + self.cfg.dice_smooth)
        # mean over classes, then batch
        return 1.0 - dice.mean()


# -----------------------------
# Example usage (minimal)
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # 3D example: (B,C,D,H,W)
    B, C, D, H, W = 2, 3, 16, 64, 64
    logits = torch.randn(B, C, D, H, W)
    target = torch.randint(low=0, high=C, size=(B, D, H, W))

    red_cfg = RedundancyConfig(
        proxy="A_prob_coherence",      # switch to B_local_moran / C_geary / D_structure_tensor / E_grad_coherence
        kernel_size_3d=(3, 7, 7),
        scalar_field="max_prob",
        norm_mode="percentile",
    )
    cfg = SRCIBConfig(
        red=red_cfg,
        lambda_ce=1.0,
        lambda_dice=1.0,
        delta_max=2.0,
        warmup_steps=1000,
        w_min=0.3,
        detach_weights=True,
        ignore_index=None,
        dice_include_background=True,
        dice_weighted=False,
    )

    loss_fn = SRCIBLoss(cfg)
    loss, stats = loss_fn(logits, target, global_step=500, return_maps=False)
    print("loss:", float(loss), "stats:", stats)