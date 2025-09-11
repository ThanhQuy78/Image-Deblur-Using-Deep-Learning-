"""utils.operators
===================

Các toán tử quan sát (A) dùng trong DIP/phục hồi ảnh:
- gaussian_kernel: sinh kernel Gaussian chuẩn hoá.
- Blur: làm mờ depthwise với kernel cố định (không train), mỗi kênh dùng cùng nhân.
- Mask: nhân mặt nạ (inpainting) với tensor đầu vào.
- DownsampleOp: giảm mẫu chống alias (dựa trên models.downsampler).
- Compose: xâu chuỗi nhiều toán tử A_n(...A_2(A_1(x))).

Cách dùng:
- Khởi tạo một hoặc nhiều toán tử và kết hợp bằng Compose để tạo A khái quát.
- Dùng trong loss DIP: tối thiểu hoá ||A(G(z;θ)) - y||^2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

try:
    # Import tuỳ chọn; chỉ dùng trong DownsampleOp
    from models.downsampler import Downsampler
except Exception:  # pragma: no cover
    Downsampler = None  # type: ignore


def gaussian_kernel(
    ks: int = 21,
    sigma: float = 2.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sinh hạt nhân Gaussian 2D đã chuẩn hoá kích thước ks x ks."""
    if ks % 2 == 0:
        raise ValueError("gaussian_kernel: ks phải là số lẻ")
    dev = device if device is not None else torch.device("cpu")
    ax = torch.arange(ks, device=dev, dtype=dtype) - (ks - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k = k / k.sum()
    return k


class Blur(nn.Module):
    """Làm mờ depthwise với nhân cố định cho mỗi kênh (không train)."""

    def __init__(
        self, kernel: torch.Tensor, channels: int = 3, padding: str = "replicate"
    ):
        super().__init__()
        if kernel.dim() == 2:
            k = kernel[None, None]  # 1x1xKxK
        elif kernel.dim() == 4 and kernel.shape[0] == 1 and kernel.shape[1] == 1:
            k = kernel
        else:
            raise ValueError("kernel must be (K,K) or (1,1,K,K)")
        k = k / (k.sum() + 1e-12)
        self.register_buffer("weight", k.repeat(channels, 1, 1, 1))  # Cx1xKxK
        pad_sz = k.shape[-1] // 2
        if padding == "replicate":
            self.pad = nn.ReplicationPad2d(pad_sz)
        elif padding == "reflect":
            self.pad = nn.ReflectionPad2d(pad_sz)
        elif padding == "zero":
            self.pad = nn.ZeroPad2d(pad_sz)
        else:
            raise ValueError(f"Kiểu padding không được hỗ trợ: {padding}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        return F.conv2d(
            x, self.weight, bias=None, stride=1, padding=0, groups=x.shape[1]
        )


class Mask(nn.Module):
    """Áp dụng mặt nạ nhị phân hoặc mềm cố định (inpainting)."""

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        if mask.dim() == 2:
            m = mask[None, None]
        elif mask.dim() == 3:
            m = mask[None]
        elif mask.dim() == 4:
            m = mask
        else:
            raise ValueError("mask must be (H,W), (C,H,W) or (1,C,H,W)")
        self.register_buffer("mask", m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Mở rộng (broadcast) mask cho khớp kênh nếu cần
        m = self.mask
        if m.shape[1] == 1 and x.shape[1] > 1:
            m = m.expand(-1, x.shape[1], -1, -1)
        return x * m


class DownsampleOp(nn.Module):
    """Giảm mẫu chống alias dựa trên models.Downsampler."""

    def __init__(
        self,
        factor: int = 2,
        kernel_type: str = "lanczos2",
        preserve_size: bool = False,
        phase: float = 0.5,
    ):
        super().__init__()
        if Downsampler is None:
            raise RuntimeError(
                "Downsampler không khả dụng. Hãy đảm bảo models/downsampler.py tồn tại và import được."
            )
        self.ds = Downsampler(
            factor=factor,
            kernel_type=kernel_type,
            preserve_size=preserve_size,
            phase=phase,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ds(x)


class Compose(nn.Module):
    """Hợp thành nhiều toán tử quan sát: y = A_n(...A_2(A_1(x)))."""

    def __init__(self, ops: Sequence[nn.Module]):
        super().__init__()
        self.ops = nn.ModuleList(list(ops))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.ops:
            x = op(x)
        return x
