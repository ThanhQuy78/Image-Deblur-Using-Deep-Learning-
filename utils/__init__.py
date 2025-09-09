"""utils package
=================
Cung cấp hàm đánh giá:
- metrics: PSNR, SSIM, evaluate_batch (định lượng).
- visualization: visualize_comparison, make_grid_triplet (định tính).
Import ví dụ: from utils import compute_psnr, visualize_comparison.
"""

from .metrics import (
    compute_psnr,
    compute_ssim,
    evaluate_batch,
)
from .visualization import (
    visualize_comparison,
    make_grid_triplet,
)

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "evaluate_batch",
    "visualize_comparison",
    "make_grid_triplet",
]
