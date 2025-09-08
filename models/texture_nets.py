"""models.texture_nets
=====================

Xây dựng mạng tổng hợp texture đa tỉ lệ (multi-scale) sử dụng trong DIP / texture synthesis.
Ý tưởng chính:
- Tạo nhiều nhánh (scale) từ ảnh đầu vào bằng Average Pool với tỉ lệ ratios[i].
- Mỗi scale: chuỗi conv (3x3,3x3,1x1) + BN + activation.
- Trộn dần các scale bằng cách: upsample output scale trước rồi concat với scale kế tiếp (Concat).
- Mỗi lần hợp nhất: tăng số kênh theo j (số scale đã gộp), rồi thêm các conv trộn lại.

Tham số:
--------
 inp           : số kênh đầu vào
 ratios        : list tỉ lệ pooling (lớn -> coarse)
 fill_noise    : nếu True chèn thêm kênh noise (GenNoise) sau pooling
 pad           : kiểu padding 'zero' | 'reflection'
 need_sigmoid  : thêm Sigmoid cuối
 conv_num      : số kênh cơ sở mỗi scale
 upsample_mode : 'nearest' | 'bilinear'

Ghi chú:
- Thiết kế ưu tiên pattern lặp / regular. Ít phù hợp trực tiếp deblur nhưng hữu dụng làm prior.
- Số kênh tăng tuyến tính theo số scale đã gộp (j) để giữ dung lượng biểu diễn.
"""

import torch.nn as nn
from .common import Concat, GenNoise, act


normalization = nn.BatchNorm2d


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad="zero"):
    """Conv helper với tùy chọn reflection padding nội bộ.

    Ghi chú: padding tính toán để giữ spatial size.
    """
    if pad == "zero":
        return nn.Conv2d(
            in_f, out_f, kernel_size, stride, padding=(kernel_size - 1) / 2, bias=bias
        )
    elif pad == "reflection":
        layers = [
            nn.ReflectionPad2d((kernel_size - 1) / 2),
            nn.Conv2d(in_f, out_f, kernel_size, stride, padding=0, bias=bias),
        ]
        return nn.Sequential(*layers)


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

        seq.add(tmp)
        if fill_noise:
            seq.add(GenNoise(inp))

        # Khối conv cơ sở cho scale i
        seq.add(conv(inp, conv_num, 3, pad=pad))
        seq.add(normalization(conv_num))
        seq.add(act())

        seq.add(conv(conv_num, conv_num, 3, pad=pad))
        seq.add(normalization(conv_num))
        seq.add(act())

        seq.add(conv(conv_num, conv_num, 1, pad=pad))
        seq.add(normalization(conv_num))
        seq.add(act())

        if i == 0:
            # Scale đầu: chỉ upsample để chuẩn bị trộn scale kế
            seq.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
            cur = seq
        else:
            cur_temp = cur  # model tích luỹ trước

            cur = nn.Sequential()

            # Batch norm trước khi merge giúp cân bằng phân phối
            seq.add(normalization(conv_num))
            cur_temp.add(normalization(conv_num * (j - 1)))

            cur.add(Concat(1, cur_temp, seq))  # concat kênh

            # Trộn lại sau concat bằng chuỗi conv
            cur.add(conv(conv_num * j, conv_num * j, 3, pad=pad))
            cur.add(normalization(conv_num * j))
            cur.add(act())

            cur.add(conv(conv_num * j, conv_num * j, 3, pad=pad))
            cur.add(normalization(conv_num * j))
            cur.add(act())

            cur.add(conv(conv_num * j, conv_num * j, 1, pad=pad))
            cur.add(normalization(conv_num * j))
            cur.add(act())

            if i == len(ratios) - 1:
                cur.add(conv(conv_num * j, output_channels, 1, pad=pad))
            else:
                cur.add(nn.Upsample(scale_factor=2, mode=upsample_mode))
    model = cur
    if output_act:
        if output_act == "sigmoid":
            model.add(nn.Sigmoid())
        elif output_act == "tanh":
            model.add(nn.Tanh())
        else:
            raise ValueError("Output_act không được hỗ trợ")
    return model
