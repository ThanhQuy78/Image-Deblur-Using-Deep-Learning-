"""models.skip
===============

Xây dựng kiến trúc encoder-decoder kiểu U-Net tùy biến với skip connections linh hoạt cho DIP.
Cho phép quy định số kênh riêng ở mỗi scale: num_channels_down, num_channels_up, num_channels_skip.

Ý tưởng:
- Mỗi scale gồm: (tùy) nhánh skip nông + nhánh deeper (downsample + conv sâu) -> merge qua Concat.
- Sau khi xử lý deeper ở scale i (và đi sâu hơn), kết thúc bằng upsample để trả feature lên trên.
- Tại điểm kết thúc của scale i: ghép skip[i] + output deeper (đã upsample) -> conv giảm kênh -> (tùy) conv 1x1 tinh chỉnh.

Tham số chính:
--------------
 num_input_channels  : kênh noise/ảnh vào
 num_output_channels : kênh ảnh ra
 num_channels_down   : list kênh encoder mỗi scale
 num_channels_up     : list kênh decoder mỗi scale
 num_channels_skip   : list kênh skip mỗi scale
 filter_size_down/up : kernel conv chính encoder/decoder
 filter_skip_size    : kernel conv nhánh skip
 upsample_mode       : 'nearest'|'bilinear'
 downsample_mode     : 'stride'|'avg'|'max'|'lanczos2'
 need1x1_up          : thêm conv 1x1 sau khối up để tăng khả năng trộn kênh
 need_sigmoid        : áp Sigmoid cuối
 pad                 : 'zero'|'reflection'
 act_fun             : chuỗi / module activation

Ghi chú:
- Việc tách cấu hình kênh giúp thử nghiệm bias kiến trúc dễ dàng (quan trọng với DIP).
- Dùng bn() sau merge để ổn định phân phối feature.
- Các nhánh deeper dùng conv stride 2 (hoặc pool theo downsample_mode) để giảm kích thước.
"""

import torch
import torch.nn as nn
from .common import Concat, bn, conv, act


def skip(
    num_input_channels=2,
    num_output_channels=3,
    num_channels_down=[16, 32, 64, 128, 128],
    num_channels_up=[16, 32, 64, 128, 128],
    num_channels_skip=[4, 4, 4, 4, 4],
    filter_size_down=3,
    filter_size_up=3,
    filter_skip_size=1,
    need_sigmoid=True,
    need_bias=True,
    pad="zero",
    upsample_mode="nearest",
    downsample_mode="stride",
    act_fun="LeakyReLU",
    need1x1_up=True,
    output_act=None,
):
    """Lắp ráp encoder-decoder nhiều scale với skip connections.

    output_act: None|'sigmoid'|'tanh'' (ưu tiên hơn need_sigmoid)
    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode = [downsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()
    model_tmp = model  # con trỏ gắn các khối deeper

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()  # nhánh deeper sẽ chứa scale tiếp theo
        skip = nn.Sequential()  # nhánh skip nông

        # Thêm Concat nếu có skip kênh >0, ngược lại chỉ deeper
        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        # Sau khi concat: chuẩn hoá số kênh bằng BN
        model_tmp.add(
            bn(
                num_channels_skip[i]
                + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])
            )
        )

        # Xây nhánh skip (nông)
        if num_channels_skip[i] != 0:
            skip.add(
                conv(
                    input_depth,
                    num_channels_skip[i],
                    filter_skip_size,
                    bias=need_bias,
                    pad=pad,
                )
            )
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))

        # Nhánh deeper: Down -> Conv giữ kênh
        deeper.add(
            conv(
                input_depth,
                num_channels_down[i],
                filter_size_down[i],
                2,
                bias=need_bias,
                pad=pad,
                downsample_mode=downsample_mode[i],
            )
        )
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(
            conv(
                num_channels_down[i],
                num_channels_down[i],
                filter_size_down[i],
                bias=need_bias,
                pad=pad,
            )
        )
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()  # placeholder cho scale sâu hơn

        if i == len(num_channels_down) - 1:
            # Deepest scale: không thêm deeper_main nữa
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]  # số kênh dự kiến trả về khi đi lên

        # Upsample trả feature lên scale hiện tại
        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        # Khối xử lý sau concat skip+deeper
        model_tmp.add(
            conv(
                num_channels_skip[i] + k,
                num_channels_up[i],
                filter_size_up[i],
                1,
                bias=need_bias,
                pad=pad,
            )
        )
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(
                conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad)
            )
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]  # cập nhật depth cho scale kế
        model_tmp = deeper_main  # đi sâu vào deeper_main để tiếp tục vòng lặp

    # Conv cuối cùng đưa về số kênh output mong muốn
    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid and output_act is None:
        output_act = "sigmoid"
    if output_act:
        if output_act == "sigmoid":
            model.add(nn.Sigmoid())
        elif output_act == "tanh":
            model.add(nn.Tanh())
        else:
            raise ValueError("Unsupported output_act")
    return model
