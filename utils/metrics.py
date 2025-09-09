"""metrics.py
================
Định lượng chất lượng ảnh khôi phục.
Bao gồm:
- compute_psnr: PSNR (độ lớn tín hiệu / nhiễu) dùng MSE.
- compute_ssim: SSIM đánh giá tương quan cấu trúc cục bộ (so sáng, tương phản, cấu trúc).
- evaluate_batch: Gộp trung bình PSNR/SSIM trên batch.

Giả định ảnh đã chuẩn hoá về [0,1] (hoặc truyền data_range nếu khác). Mọi tensor sẽ được
ép sang float và broadcast kích thước bằng nội suy bilinear nếu lệch.
"""

import math
from typing import Union, Sequence, Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["compute_psnr", "compute_ssim", "evaluate_batch"]

# ---- helpers ----


def _to_tensor(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x)


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:  # H W
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:  # C H W
        x = x.unsqueeze(0)
    return x


def _match_shapes(
    a: torch.Tensor, b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
    return a, b


# ---- PSNR ----


def compute_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
) -> float:
    pred_t = _ensure_4d(_to_tensor(pred).float())
    tgt_t = _ensure_4d(_to_tensor(target).float())
    pred_t, tgt_t = _match_shapes(pred_t, tgt_t)
    mse = F.mse_loss(pred_t, tgt_t).item()
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


# ---- SSIM ----


def _gaussian_window(
    window_size: int = 11, sigma: float = 1.5, channels: int = 3
) -> torch.Tensor:
    coords = torch.arange(window_size).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.t() @ g
    window = window_2d.unsqueeze(0).unsqueeze(0)
    window = window.repeat(channels, 1, 1, 1)
    return window


def compute_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-8,
) -> float:
    pred_t = _ensure_4d(_to_tensor(pred).float())
    tgt_t = _ensure_4d(_to_tensor(target).float())
    pred_t, tgt_t = _match_shapes(pred_t, tgt_t)
    C, H, W = pred_t.shape[1:]
    if window_size > min(H, W):
        window_size = min(H, W)
        if window_size % 2 == 0:
            window_size -= 1
    window = _gaussian_window(window_size, sigma, channels=pred_t.shape[1]).to(
        pred_t.device, pred_t.dtype
    )
    pad = window_size // 2
    mu_x = F.conv2d(pred_t, window, groups=pred_t.shape[1], padding=pad)
    mu_y = F.conv2d(tgt_t, window, groups=tgt_t.shape[1], padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = (
        F.conv2d(pred_t * pred_t, window, groups=pred_t.shape[1], padding=pad) - mu_x2
    )
    sigma_y2 = (
        F.conv2d(tgt_t * tgt_t, window, groups=tgt_t.shape[1], padding=pad) - mu_y2
    )
    sigma_xy = (
        F.conv2d(pred_t * tgt_t, window, groups=pred_t.shape[1], padding=pad) - mu_xy
    )
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps
    )
    return ssim_map.mean().item()


# ---- Batch evaluation ----


def evaluate_batch(
    pred_batch: torch.Tensor,
    target_batch: torch.Tensor,
    data_range: float = 1.0,
    metrics: Sequence[str] = ("psnr", "ssim"),
) -> Dict[str, float]:
    pred_batch = pred_batch.detach().float()
    target_batch = target_batch.detach().float()
    if pred_batch.shape != target_batch.shape:
        raise ValueError("pred_batch và target_batch phải cùng shape")
    B = pred_batch.shape[0]
    results = {m: 0.0 for m in metrics}
    for i in range(B):
        p = pred_batch[i : i + 1]
        t = target_batch[i : i + 1]
        if "psnr" in metrics:
            results["psnr"] += compute_psnr(p, t, data_range)
        if "ssim" in metrics:
            results["ssim"] += compute_ssim(p, t, data_range)
    for k in results:
        results[k] /= B
    return results
