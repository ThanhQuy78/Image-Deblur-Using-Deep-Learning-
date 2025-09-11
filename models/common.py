"""models.common
================

Tiện ích dùng chung để lắp ráp các kiến trúc DIP:
- Concat: nối feature từ nhiều nhánh, tự crop trung tâm nếu lệch kích thước.
- GenNoise: sinh noise Gaussian dạng feature map với số kênh tuỳ chọn.
- Swish/act: factory kích hoạt.
- bn: BatchNorm2d cơ bản.
- conv: bọc Conv2d với padding 'reflection' và cơ chế downsample rõ ràng (avg/max/lanczos) tách khỏi stride conv.
- weight_init_*: các hàm khởi tạo trọng số phổ biến.

Lưu ý tương thích:
- Hàm conv sẽ dùng Downsampler khi downsample_mode in {'lanczos2','lanczos3'} để chống alias.
- Monkey patch Module.add giữ lại để không phá code cũ (không khuyến nghị dùng trong code mới).
"""

import torch
import torch.nn as nn
from .downsampler import Downsampler

__all__ = [
    "Concat",
    "GenNoise",
    "Swish",
    "act",
    "bn",
    "conv",
    "weight_init_dcgan",
    "weight_init_kaiming",
]


# -----------------------------------------------------------------------------
# Monkey patch lịch sử (giữ để tương thích; KHÔNG khuyến nghị dùng mới)
# -----------------------------------------------------------------------------
def add_module(self, module):  # pragma: no cover
    self.add_module(str(len(self) + 1), module)


torch.nn.Module.add = add_module  # type: ignore


# -----------------------------------------------------------------------------
# Core building blocks
# -----------------------------------------------------------------------------
# Concat: Ghép tensor theo chiều dim, tự canh/crop trung tâm nếu lệch 1-2 px do upsample/pooling.
class Concat(nn.Module):
    """Ghép output nhiều module theo dim; auto crop nếu lệch kích thước.

    Lưu ý: Crop trung tâm -> giả định lệch do rounding/upsample; giữ nội dung quan trọng.
    """

    def __init__(self, dim, *args):
        super().__init__()
        self.dim = dim
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        # Thu thập output các nhánh và cắt đồng kích thước theo tâm trước khi concat
        outputs = [m(input) for m in self._modules.values()]
        h_targets = [o.shape[2] for o in outputs]
        w_targets = [o.shape[3] for o in outputs]
        tgt_h = min(h_targets)
        tgt_w = min(w_targets)
        if not (
            all(h == tgt_h for h in h_targets) and all(w == tgt_w for w in w_targets)
        ):
            cropped = []
            for o in outputs:
                dh = (o.size(2) - tgt_h) // 2
                dw = (o.size(3) - tgt_w) // 2
                cropped.append(o[:, :, dh : dh + tgt_h, dw : dw + tgt_w])
            outputs = cropped
        return torch.cat(outputs, dim=self.dim)

    def __len__(self):  # pragma: no cover
        return len(self._modules)


# GenNoise: Sinh tensor noise chuẩn N(0,1) mới cho mỗi lần forward (trên cùng device/dtype với input).
class GenNoise(nn.Module):
    """Sinh noise Gaussian mới mỗi forward (không cố định seed)."""

    def __init__(self, dim2):
        super().__init__()
        self.dim2 = dim2

    def forward(self, input):
        # shape out có cùng B,H,W với input, nhưng số kênh thay bằng dim2
        shape = list(input.size())
        shape[1] = self.dim2
        noise = torch.randn(shape, device=input.device, dtype=input.dtype)
        return noise


# Swish: x * sigmoid(x)
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Sigmoid()

    def forward(self, x):  # pragma: no cover (đơn giản)
        return x * self.sig(x)


def act(act_fun="LeakyReLU"):
    """Factory trả về module kích hoạt.

    Hỗ trợ chuỗi: 'LeakyReLU' | 'Swish' | 'ELU' | 'none';
    Hoặc truyền vào class module tuỳ chỉnh -> sẽ được khởi tạo không tham số.
    """
    if isinstance(act_fun, str):
        if act_fun == "LeakyReLU":
            return nn.LeakyReLU(0.2, inplace=True)
        if act_fun == "Swish":
            return Swish()
        if act_fun == "ELU":
            return nn.ELU(inplace=True)
        if act_fun == "none":
            return nn.Identity()
        raise ValueError(f"Unsupported activation: {act_fun}")
    return act_fun()


def bn(num_features):
    """BatchNorm2d với affine=True (theo mặc định của PyTorch)."""
    return nn.BatchNorm2d(num_features)


def conv(
    in_f: int,
    out_f: int,
    kernel_size: int,
    stride: int = 1,
    bias: bool = True,
    pad: str = "zero",
    downsample_mode: str = "stride",
) -> nn.Module:
    """Conv2d wrapper với padding 'reflection' và downsample rõ ràng.

    Tham số:
    - in_f/out_f: số kênh vào/ra
    - kernel_size: kích thước kernel (số lẻ)
    - stride: nếu >1 & downsample_mode=='stride' -> dùng conv stride; ngược lại dùng Avg/Max/Downsampler
    - pad: 'zero' | 'reflection'
    - downsample_mode: 'stride' | 'avg' | 'max' | 'lanczos2' | 'lanczos3'
    """
    downsampler = None
    if stride != 1 and downsample_mode != "stride":
        if downsample_mode == "avg":
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == "max":
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ("lanczos2", "lanczos3"):
            downsampler = Downsampler(
                n_planes=out_f,
                factor=stride,
                kernel_type=downsample_mode,
                phase=0.5,
                preserve_size=True,
            )
        else:
            raise ValueError(f"Unsupported downsample_mode: {downsample_mode}")
        stride = 1  # conv stride=1 sau khi tách downsample

    pad_layer = None
    pad_amt = (kernel_size - 1) // 2
    if pad == "reflection":
        pad_layer = nn.ReflectionPad2d(pad_amt)
        pad_amt = 0
    conv_layer = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=pad_amt, bias=bias)

    layers = []
    if pad_layer is not None:
        layers.append(pad_layer)
    layers.append(conv_layer)
    if downsampler is not None:
        layers.append(downsampler)
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Weight init utilities
# -----------------------------------------------------------------------------
def weight_init_dcgan(m):  # pragma: no cover (thường gọi qua .apply)
    """Khởi tạo kiểu DCGAN: Conv ~ N(0,0.02), BN: gamma~N(1,0.02), beta=0."""
    cls = m.__class__.__name__
    if cls.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif cls.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weight_init_kaiming(m):  # pragma: no cover
    """Khởi tạo Kaiming cho Conv2d (leaky_relu), BN: gamma=1, beta=0."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
