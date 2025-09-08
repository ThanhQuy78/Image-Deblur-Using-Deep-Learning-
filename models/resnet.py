"""models.resnet
=================

Triển khai mạng ResNet đơn giản dùng cho Deep Image Prior / phục hồi ảnh.
Cấu trúc: Conv đầu -> chuỗi residual blocks (Conv-BN-Act-Conv-BN) -> Conv + BN -> Conv cuối -> Sigmoid (tuỳ chọn).

Tham số chính (ResNet class):
-----------------------------
 num_input_channels : kênh đầu vào (noise/ảnh degraded)
 num_output_channels: kênh đầu ra (ảnh khôi phục)
 num_blocks         : số residual blocks (độ sâu phần giữa)
 num_channels       : số kênh feature cố định toàn bộ mạng
 need_residual      : True -> dùng ResidualSequential bọc block (tự cộng skip); False -> stack tuần tự
 act_fun            : hàm kích hoạt nội bộ ('LeakyReLU', 'ELU', ...)
 need_sigmoid       : True -> áp sigmoid cuối (giới hạn [0,1])
 norm_layer         : lớp chuẩn hoá (BatchNorm2d / InstanceNorm2d ...)
 pad                : kiểu padding cho conv helper ('reflection' | 'zero')

Ghi chú thiết kế:
-----------------
* Các block giữ nguyên spatial size (padding=1 cho kernel 3).
* ResidualSequential hiện tự cắt crop nếu kích thước lệch (phòng hộ, hiếm khi xảy ra).
* Chưa có cơ chế down/up-sampling nên depth ảnh hưởng chủ yếu đến biểu diễn phi tuyến.
* Có thể mở rộng: thêm dilated conv, weight init tùy chỉnh, bỏ Sigmoid nếu training với range khác.
"""

import torch.nn as nn
from .common import act, conv


class ResidualSequential(nn.Sequential):
    """Chuỗi layer với skip connection tổng.

    Forward: out = F(x); nếu spatial không khớp sẽ crop x để cộng.
    """

    def __init__(self, *args):
        super(ResidualSequential, self).__init__(*args)

    def forward(self, x):
        out = super(ResidualSequential, self).forward(x)
        if out.size(2) != x.size(2) or out.size(3) != x.size(3):
            diff2 = x.size(2) - out.size(2)
            diff3 = x.size(3) - out.size(3)
            x = x[
                :,
                :,
                diff2 // 2 : diff2 // 2 + out.size(2),
                diff3 // 2 : diff3 // 2 + out.size(3),
            ]
        return out + x


def get_block(num_channels, norm_layer, act_fun):
    """Tạo 1 residual block: Conv -> BN -> Act -> Conv -> BN (chưa cộng skip)."""
    layers = [
        nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False),
        norm_layer(num_channels, affine=True),
        act(act_fun),
        nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=False),
        norm_layer(num_channels, affine=True),
    ]
    return layers


class ResNet(nn.Module):
    """ResNet đơn giản cho DIP.

    Pipeline:
    1. Conv đầu (conv helper hỗ trợ padding reflection/zero)
    2. Chuỗi residual blocks (ResidualSequential hoặc Sequential)
    3. Conv + BN ổn định feature
    4. Conv cuối -> (Sigmoid nếu need_sigmoid)
    """

    def __init__(
        self,
        num_input_channels,
        num_output_channels,
        num_blocks,
        num_channels,
        need_residual=True,
        act_fun="LeakyReLU",
        need_sigmoid=True,
        norm_layer=nn.BatchNorm2d,
        pad="reflection",
        output_act=None,
    ):
        """Khởi tạo ResNet.

        pad: 'start|zero|replication' (giữ lại note gốc, thực tế conv helper hỗ trợ 'zero' | 'reflection').
        """
        super(ResNet, self).__init__()

        if need_residual:
            s = ResidualSequential
        else:
            s = nn.Sequential

        # First layers
        layers = [
            conv(num_input_channels, num_channels, 3, stride=1, bias=True, pad=pad),
            act(act_fun),
        ]
        # Residual blocks
        for i in range(num_blocks):
            layers += [s(*get_block(num_channels, norm_layer, act_fun))]

        # Transition Conv + BN
        layers += [
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            norm_layer(num_channels, affine=True),
        ]

        # Output conv + activation
        tail_act = nn.Identity()
        if output_act is not None:
            if output_act == "sigmoid":
                tail_act = nn.Sigmoid()
            elif output_act == "tanh":
                tail_act = nn.Tanh()
            else:
                raise ValueError("Unsupported output_act")
        elif need_sigmoid:
            tail_act = nn.Sigmoid()

        layers += [
            conv(num_channels, num_output_channels, 3, 1, bias=True, pad=pad),
            tail_act,
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

    def eval(self):
        return super().eval()
