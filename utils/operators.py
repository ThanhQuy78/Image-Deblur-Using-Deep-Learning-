"""utils.operators
===================

Các toán tử quan sát (A) dùng trong DIP/phục hồi ảnh:
- gaussian_kernel: sinh kernel Gaussian chuẩn hoá.
- Blur: làm mờ depthwise với kernel cố định (không train), mỗi kênh dùng cùng nhân.
- MotionBlur: làm mờ theo hướng (độ dài, góc) mô phỏng chuyển động tuyến tính.
- PiecewiseBlur: mờ biến thiên theo không gian theo lưới ô, trộn mép mượt (phù hợp blur không đồng nhất như GoPro).
- Mask: nhân mặt nạ (inpainting) với tensor đầu vào.
- DownsampleOp: giảm mẫu chống alias (dựa trên models.downsampler).
- Compose: xâu chuỗi nhiều toán tử A_n(...A_2(A_1(x))).

Cách dùng:
- Khởi tạo một hoặc nhiều toán tử và kết hợp bằng Compose để tạo A khái quát.
- Dùng trong loss DIP: tối thiểu hoá ||A(G(z;θ)) - y||^2.
"""

import math
from typing import Optional, Sequence, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    k = k / (k.sum() + 1e-12)
    return k


def motion_kernel(
    length: int = 9,
    angle_deg: float = 0.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sinh hạt nhân mờ chuyển động tuyến tính (đường thẳng) với độ dài và góc cho trước.

    Ghi chú: kernel được vẽ trên lưới vuông kích thước length x length.
    """
    if length < 1:
        raise ValueError("motion_kernel: length phải >= 1")
    ks = length if length % 2 == 1 else length + 1
    dev = device if device is not None else torch.device("cpu")
    k = torch.zeros((ks, ks), device=dev, dtype=dtype)
    c = ks // 2
    theta = math.radians(angle_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)
    # Vẽ đường qua tâm theo góc theta
    num = 0
    for t in torch.linspace(-c, c, steps=ks, device=dev):
        x = c + t * dx
        y = c + t * dy
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if 0 <= xi < ks and 0 <= yi < ks:
            k[yi, xi] = 1.0
            num += 1
    if num == 0:
        k[c, c] = 1.0
        num = 1
    k = k / k.sum()
    return k


class Blur(nn.Module):
    """Làm mờ depthwise với nhân cố định cho mỗi kênh (không train)."""

    def __init__(
        self, kernel: torch.Tensor, channels: int = 3, padding: str = "replicate"
    ):
        super().__init__()
        if kernel.dim() == 2:
            k = kernel[None, None]
        elif kernel.dim() == 4 and kernel.shape[0] == 1 and kernel.shape[1] == 1:
            k = kernel
        else:
            raise ValueError("kernel phải có dạng (K,K) hoặc (1,1,K,K)")
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


class MotionBlur(Blur):
    """Làm mờ chuyển động theo hướng (độ dài, góc) như một trường hợp đặc biệt của Blur."""

    def __init__(
        self,
        length: int = 9,
        angle_deg: float = 0.0,
        channels: int = 3,
        padding: str = "replicate",
        device: Optional[torch.device] = None,
    ):
        k = motion_kernel(length=length, angle_deg=angle_deg, device=device)
        super().__init__(k, channels=channels, padding=padding)


class PiecewiseBlur(nn.Module):
    """Mờ biến thiên theo không gian theo từng ô (tile) với trộn mép mượt.

    - Chia ảnh thành lưới (ny x nx). Mỗi ô có 1 kernel (KxK) dùng cho mọi kênh.
    - Trộn mép bằng cửa sổ cos để tránh đường nối giữa các ô.
    - Kernel truyền vào dưới dạng mảng 2 chiều (ny x nx) các tensor.
    """

    def __init__(
        self,
        kernels: Sequence[Sequence[torch.Tensor]],
        grid_size: Tuple[int, int],
        channels: int = 3,
        padding: str = "replicate",
        blend: bool = True,
        blend_ratio: float = 0.15,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        ny, nx = grid_size
        assert len(kernels) == ny and all(len(row) == nx for row in kernels), (
            "PiecewiseBlur: số kernel phải khớp grid_size (ny x nx)"
        )
        self.ny, self.nx = ny, nx
        self.padding = padding
        self.blend = bool(blend)
        self.blend_ratio = float(blend_ratio)
        self.channels = int(channels)

        # Lưu các toán tử Blur theo từng ô để tự động move device
        rows: List[nn.ModuleList] = []
        for j in range(ny):
            row_ops: List[nn.Module] = []
            for i in range(nx):
                k = kernels[j][i]
                if device is not None:
                    k = k.to(device)
                row_ops.append(Blur(k, channels=channels, padding=padding))
            rows.append(nn.ModuleList(row_ops))
        self.ops = nn.ModuleList(rows)

    @staticmethod
    def _cosine_window(h: int, w: int, br: float, device, dtype):
        """Cửa sổ cos 2D (0..1) làm trọng số trộn mép cho một ô."""
        yh = max(1, int(round(h * br)))
        xw = max(1, int(round(w * br)))
        y = torch.ones(h, 1, device=device, dtype=dtype)
        if yh > 0:
            t = torch.linspace(0, math.pi, steps=yh, device=device, dtype=dtype)
            fade = 0.5 * (1 - torch.cos(t))
            y[:yh, 0] = fade
            y[-yh:, 0] = fade.flip(0)
        x = torch.ones(1, w, device=device, dtype=dtype)
        if xw > 0:
            t = torch.linspace(0, math.pi, steps=xw, device=device, dtype=dtype)
            fade = 0.5 * (1 - torch.cos(t))
            x[0, :xw] = fade
            x[0, -xw:] = fade.flip(1)
        return (y @ x).clamp_(1e-6, 1.0)  # tránh chia 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, H, W = x.shape
        ny, nx = self.ny, self.nx
        # Kích thước mỗi ô (chia gần đều)
        hh = [H // ny + (1 if y < H % ny else 0) for y in range(ny)]
        ww = [W // nx + (1 if x_ < W % nx else 0) for x_ in range(nx)]
        y0 = [0]
        for h in hh[:-1]:
            y0.append(y0[-1] + h)
        x0 = [0]
        for w in ww[:-1]:
            x0.append(x0[-1] + w)

        out = torch.zeros_like(x)
        acc = (
            torch.zeros((b, 1, H, W), device=x.device, dtype=x.dtype)
            if self.blend
            else None
        )

        for j in range(ny):
            for i in range(nx):
                y1 = y0[j] + hh[j]
                x1 = x0[i] + ww[i]
                patch = x[:, :, y0[j] : y1, x0[i] : x1]
                y_hat = self.ops[j][i](patch)
                if self.blend:
                    wmask = self._cosine_window(
                        hh[j], ww[i], self.blend_ratio, x.device, x.dtype
                    )
                    wmask = wmask[None, None].expand(b, 1, hh[j], ww[i])
                    out[:, :, y0[j] : y1, x0[i] : x1] += y_hat * wmask
                    acc[:, :, y0[j] : y1, x0[i] : x1] += wmask
                else:
                    out[:, :, y0[j] : y1, x0[i] : x1] = y_hat

        if self.blend:
            out = out / acc.clamp_min(1e-6)
        return out


class Mask(nn.Module):
    """Áp dụng mặt nạ nhị phân hoặc mềm cố định (inpainting). y = M ⊙ x."""

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        if mask.dim() == 2:
            m = mask[None, None]
        elif mask.dim() == 3:
            m = mask[None]
        elif mask.dim() == 4:
            m = mask
        else:
            raise ValueError("mask phải là (H,W), (C,H,W) hoặc (1,C,H,W)")
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
