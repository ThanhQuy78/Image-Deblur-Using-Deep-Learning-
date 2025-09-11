"""models.texture_nets
=====================

Mạng tổng hợp texture đa tỉ lệ dùng trong DIP/texture synthesis.
- Mỗi scale: AvgPool -> chuỗi Conv(3,3,1) + BN + Act.
- Gộp dần: Upsample output scale trước, Concat với scale hiện tại, rồi Conv trộn lại.
- Hỗ trợ tuỳ chọn noise nhánh (fill_noise) để tăng đa dạng hoá pattern.

Tham số chính: inp, ratios, fill_noise, pad, conv_num, upsample_mode, output_channels, output_act.

Ghi chú:
- Thiết kế ưu tiên pattern lặp / regular. Ít phù hợp trực tiếp deblur nhưng hữu dụng làm prior.
- Số kênh tăng tuyến tính theo số scale đã gộp (j) để giữ dung lượng biểu diễn.
"""

import torch.nn as nn
from .common import Concat, GenNoise, act


normalization = nn.BatchNorm2d


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad="zero"):
    """Conv helper với tùy chọn reflection padding nội bộ.

    Ghi chú: padding tính toán để giữ spatial size (ép kiểu int).
    """
    pad_val = (kernel_size - 1) // 2  # đảm bảo int
    if pad == "zero":
        return nn.Conv2d(in_f, out_f, kernel_size, stride, padding=pad_val, bias=bias)
    elif pad == "reflection":
        layers = [
            nn.ReflectionPad2d(pad_val),
            nn.Conv2d(in_f, out_f, kernel_size, stride, padding=0, bias=bias),
        ]
        return nn.Sequential(*layers)


# Helper: add to Sequential with unique names
_counter = {"seq": 0}


def add(m: nn.Sequential, layer: nn.Module, prefix: str = "layer"):
    _counter["seq"] += 1
    m.add_module(f"{prefix}_{_counter['seq']}", layer)


def get_texture_nets(
    inp=3,
    ratios=[32, 16, 8, 4, 2, 1],
    fill_noise=False,
    pad="zero",
    need_sigmoid=False,
    conv_num=8,
    upsample_mode="nearest",
    output_channels=3,
    output_act=None,
):
    """Lắp ráp mạng tổng hợp texture đa tỉ lệ.

    Quy trình cho mỗi scale i:
    1. AvgPool2d(ratios[i]) lấy bản coarse
    2. (Tuỳ chọn) GenNoise thêm kênh nhiễu
    3. Chuỗi conv (3x3 -> 3x3 -> 1x1) + BN + act
    4. Nếu không phải scale đầu: hợp nhất với kết quả tích luỹ trước (Concat)
         - BN riêng từng nhánh trước khi concat (ổn định thống kê)
         - Sau concat: conv 3x3 -> conv 3x3 -> conv 1x1 (giữ kênh conv_num * j)
    5. Upsample (trừ scale cuối) đưa về kích thước gần scale tiếp theo

    Trả về: model (nn.Sequential) có thể thêm Sigmoid nếu need_sigmoid.
    """
    for i in range(len(ratios)):
        j = i + 1  # số scale đã gộp

        seq = nn.Sequential()

        tmp = nn.AvgPool2d(ratios[i], ratios[i])  # pooling coarse

        add(seq, tmp, "pool")
        if fill_noise:
            add(seq, GenNoise(inp), "noise")

        # Khối conv cơ sở cho scale i
        add(seq, conv(inp, conv_num, 3, pad=pad), "conv")
        add(seq, normalization(conv_num), "bn")
        add(seq, act(), "act")

        add(seq, conv(conv_num, conv_num, 3, pad=pad), "conv")
        add(seq, normalization(conv_num), "bn")
        add(seq, act(), "act")

        add(seq, conv(conv_num, conv_num, 1, pad=pad), "conv")
        add(seq, normalization(conv_num), "bn")
        add(seq, act(), "act")

        if i == 0:
            # Scale đầu: chỉ upsample để chuẩn bị trộn scale kế
            add(seq, nn.Upsample(scale_factor=2, mode=upsample_mode), "ups")
            cur = seq
        else:
            cur_temp = cur  # model tích luỹ trước

            cur = nn.Sequential()

            # Batch norm trước khi merge giúp cân bằng phân phối
            add(seq, normalization(conv_num), "bn")
            add(cur_temp, normalization(conv_num * (j - 1)), "bn")

            add(cur, Concat(1, cur_temp, seq), "concat")  # concat kênh

            # Trộn lại sau concat bằng chuỗi conv
            add(cur, conv(conv_num * j, conv_num * j, 3, pad=pad), "conv")
            add(cur, normalization(conv_num * j), "bn")
            add(cur, act(), "act")

            add(cur, conv(conv_num * j, conv_num * j, 3, pad=pad), "conv")
            add(cur, normalization(conv_num * j), "bn")
            add(cur, act(), "act")

            add(cur, conv(conv_num * j, conv_num * j, 1, pad=pad), "conv")
            add(cur, normalization(conv_num * j), "bn")
            add(cur, act(), "act")

            if i == len(ratios) - 1:
                add(cur, conv(conv_num * j, output_channels, 1, pad=pad), "head")
            else:
                add(cur, nn.Upsample(scale_factor=2, mode=upsample_mode), "ups")
    model = cur
    if output_act:
        if output_act == "sigmoid":
            add(model, nn.Sigmoid(), "sigmoid")
        elif output_act == "tanh":
            add(model, nn.Tanh(), "tanh")
        else:
            raise ValueError("Output_act không được hỗ trợ")
    return model
