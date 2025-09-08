"""models.common
================

Helper chung cho các kiến trúc DIP:
- Concat: nối feature nhiều nhánh, auto crop trung tâm nếu lệch kích thước.
- GenNoise: sinh noise Gaussian dạng feature map có kênh tùy chọn.
- Swish / act: factory kích hoạt.
- bn: BatchNorm2d đơn giản.
- conv: wrapper cho Conv2d kèm tuỳ chọn downsample rõ ràng (avg/max/lanczos) thay stride conv.
- Hàm khởi tạo trọng số phổ biến (DCGAN style / Kaiming) để dễ apply từ ngoài.

Tối ưu/Chỉnh sửa:
- Thêm __all__ để giới hạn export.
- conv(): bỏ filter() để giảm overhead, code rõ ràng hơn.
- Đưa kernel downsample (Lanczos) sang Downsampler (đã register_buffer ở file kia) -> an toàn device.
- Cảnh báo: Monkey patch Module.add vẫn giữ để không phá code cũ nhưng không khuyến khích dùng.
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


class GenNoise(nn.Module):
    """Sinh noise Gaussian mới mỗi forward (không cố định seed)."""

    def __init__(self, dim2):
        super().__init__()
        self.dim2 = dim2

    def forward(self, input):
        shape = list(input.size())
        shape[1] = self.dim2
        noise = torch.randn(shape, device=input.device, dtype=input.dtype)
        return noise


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Sigmoid()

    def forward(self, x):  # pragma: no cover (đơn giản)
        return x * self.sig(x)


def act(act_fun="LeakyReLU"):
    """Factory trả về activation.

    Hỗ trợ: 'LeakyReLU' | 'Swish' | 'ELU' | 'none'.
    Có thể truyền module class tùy chỉnh -> sẽ được khởi tạo không tham số.
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
    """Wrapper conv với tuỳ chọn downsample rõ ràng & reflection pad.

    Args:
        in_f/out_f: kênh in/out
        kernel_size: kích thước kernel (odd)
        stride: nếu >1 & downsample_mode=='stride' -> conv stride; else dùng pooling/Downsampler.
        pad: 'zero' | 'reflection'
        downsample_mode: 'stride' | 'avg' | 'max' | 'lanczos2' | 'lanczos3'
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
    cls = m.__class__.__name__
    if cls.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif cls.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weight_init_kaiming(m):  # pragma: no cover
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
