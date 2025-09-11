"""models.dcgan
================

Generator DCGAN tối giản dùng trong vai trò prior (ví dụ DIP) hoặc ví dụ sinh ảnh.
Tuỳ chọn hai chế độ upsampling:
- ConvTranspose2d (need_convT=True)
- Upsample + Conv (need_convT=False) để giảm checkerboard.

Tham số:
- inp: số kênh đầu vào (noise map)
- ndf: số kênh ẩn cố định (feature width)
- num_ups: số khối phóng to (bao gồm block cuối)
- need_sigmoid: True -> thêm Sigmoid nếu output_act không chỉ định
- need_bias: bias cho Conv/ConvTranspose
- pad: reserved (hiện chưa dùng mở rộng padding)
- upsample_mode: 'nearest'|'bilinear' cho nhánh Upsample+Conv
- need_convT: True -> dùng ConvTranspose2d, False -> Upsample+Conv
- output_channels: số kênh ảnh đầu ra (mặc định 3)
- output_act: None|'sigmoid'|'tanh' (override need_sigmoid)
- weight_init: None|'dcgan'|'kaiming'

Gợi ý:
- DIP: giữ noise đầu vào cố định trong suốt tối ưu.
- Tránh checkerboard: ưu tiên Upsample+Conv.
"""

import torch.nn as nn
from .common import weight_init_dcgan, weight_init_kaiming


def dcgan(
    inp=2,
    ndf=32,
    num_ups=4,
    need_sigmoid=True,
    need_bias=True,
    pad="zero",
    upsample_mode="nearest",
    need_convT=True,
    output_channels=3,
    output_act=None,
    weight_init=None,
):
    """Tạo generator DCGAN tối giản.

    output_act ưu tiên hơn need_sigmoid.
    weight_init: None|'dcgan'|'kaiming'.
    """
    # Xác định activation cuối
    if need_sigmoid and output_act is None:
        output_act = "sigmoid"

    layers = [
        nn.ConvTranspose2d(
            inp, ndf, kernel_size=3, stride=1, padding=0, bias=need_bias
        ),
        nn.BatchNorm2d(ndf),
        nn.LeakyReLU(0.2, inplace=True),
    ]

    # Các tầng phóng to trung gian
    for _ in range(max(0, num_ups - 3)):
        if need_convT:
            layers += [
                nn.ConvTranspose2d(
                    ndf, ndf, kernel_size=4, stride=2, padding=1, bias=need_bias
                ),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        else:
            layers += [
                nn.Upsample(scale_factor=2, mode=upsample_mode),
                nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=need_bias),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
            ]

    # Tầng cuối ra ảnh
    if need_convT:
        layers += [nn.ConvTranspose2d(ndf, output_channels, 4, 2, 1, bias=need_bias)]
    else:
        layers += [
            nn.Upsample(scale_factor=2, mode=upsample_mode),
            nn.Conv2d(
                ndf, output_channels, kernel_size=3, stride=1, padding=1, bias=need_bias
            ),
        ]

    # Activation cuối
    if output_act:
        if output_act == "sigmoid":
            layers += [nn.Sigmoid()]
        elif output_act == "tanh":
            layers += [nn.Tanh()]
        else:
            raise ValueError("Unsupported output_act")

    model = nn.Sequential(*layers)

    # Khởi tạo trọng số nếu yêu cầu
    if weight_init:
        if weight_init == "dcgan":
            model.apply(weight_init_dcgan)
        elif weight_init == "kaiming":
            model.apply(weight_init_kaiming)
        else:
            raise ValueError("Unsupported weight_init")

    return model
