"""models.__init__
===================

Factory cung cấp hàm get_net khởi tạo kiến trúc mạng cho DIP / phục hồi ảnh.

Các kiến trúc hỗ trợ:
1. 'skip'          : Encoder–decoder nhiều tầng + skip tùy biến kênh.
2. 'texture_nets'  : Tổng hợp texture đa tỉ lệ.
3. 'ResNet'        : Chuỗi residual blocks chiều sâu vừa.
4. 'UNet'          : U-Net chuẩn với tuỳ chọn feature_scale.
5. 'dcgan'         : DCGAN generator tối giản làm image prior.
6. 'identity'      : Trả về nn.Sequential() rỗng (debug/baseline, yêu cầu input_depth==n_channels).

output_act (global): None|'sigmoid'|'tanh' ưu tiên override need_sigmoid.
"""

from .skip import skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet
from .dcgan import dcgan
import torch.nn as nn

__all__ = ["get_net", "skip", "get_texture_nets", "ResNet", "UNet", "dcgan"]


def get_net(
    NET_TYPE,
    input_depth,
    pad="reflection",
    upsample_mode="bilinear",
    n_channels=3,
    act_fun="LeakyReLU",
    skip_n33d=128,
    skip_n33u=128,
    skip_n11=4,
    num_scales=5,
    downsample_mode="stride",
    need_sigmoid=True,
    res_blocks=8,
    res_feat=64,
    norm_layer=nn.BatchNorm2d,
    output_act=None,
    dcgan_ndf=64,
    dcgan_ups=4,
    dcgan_convT=True,
):
    """Trả về instance mạng theo ``NET_TYPE``.

    output_act: None|'sigmoid'|'tanh' (override need_sigmoid nếu cung cấp)
    """
    if NET_TYPE == "ResNet":
        # ResNet: chuỗi các residual blocks. Ổn định gradient, phù hợp khi không cần
        # kiến trúc encoder–decoder đa tỉ lệ nhưng vẫn muốn biểu diễn sâu vừa phải.
        net = ResNet(
            input_depth,
            n_channels,
            num_blocks=res_blocks,
            num_channels=res_feat,
            need_sigmoid=int(need_sigmoid),
            norm_layer=norm_layer,
            need_bias=True,
            output_act=output_act,
        )
    elif NET_TYPE == "skip":
        # Skip network: linh hoạt cấu hình số kênh mỗi scale. Tạo mạnh inductive bias
        # về cấu trúc ảnh tự nhiên và thường đạt chất lượng cao trong DIP.
        net = skip(
            input_depth,
            n_channels,
            num_channels_down=[skip_n33d] * num_scales
            if isinstance(skip_n33d, int)
            else skip_n33d,
            num_channels_up=[skip_n33u] * num_scales
            if isinstance(skip_n33u, int)
            else skip_n33u,
            num_channels_skip=[skip_n11] * num_scales
            if isinstance(skip_n11, int)
            else skip_n11,
            upsample_mode=upsample_mode,
            downsample_mode=downsample_mode,
            need_sigmoid=need_sigmoid,
            need_bias=True,
            pad=pad,
            act_fun=act_fun,
            output_act=output_act,
        )
    elif NET_TYPE == "texture_nets":
        # Texture nets: multi-scale feature synthesis. Thường dùng cho bài toán tạo / giữ pattern.
        net = get_texture_nets(
            inp=input_depth, ratios=[32, 16, 8, 4, 2, 1], fill_noise=False, pad=pad
        )
    elif NET_TYPE == "UNet":
        # UNet: kiến trúc đối xứng cổ điển. Cấu hình hiện tại chọn feature_scale=4 để
        # giảm số kênh gốc (tiết kiệm bộ nhớ) – có thể điều chỉnh tùy dataset.
        net = UNet(
            num_input_channels=input_depth,
            num_output_channels=n_channels,
            feature_scale=4,
            more_layers=0,
            concat_x=False,
            upsample_mode=upsample_mode,
            pad=pad,
            norm_layer=norm_layer,
            need_sigmoid=need_sigmoid,
            need_bias=True,
            final_activation=output_act,
        )
    elif NET_TYPE == "dcgan":
        # DCGAN: mạng sinh DCGAN tối giản, chủ yếu dùng cho bài toán tạo ảnh.
        net = dcgan(
            inp=input_depth,
            ndf=dcgan_ndf,
            num_ups=dcgan_ups,
            need_sigmoid=need_sigmoid,
            upsample_mode=upsample_mode,
            need_convT=dcgan_convT,
            output_channels=n_channels,
            output_act=output_act,
        )
    elif NET_TYPE == "identity":
        # Identity: dùng chủ yếu để debug pipeline hoặc làm baseline (không học gì).
        assert input_depth == n_channels, "identity: input_depth phải bằng n_channels"
        net = nn.Sequential()
    else:
        raise ValueError(f"Unknown NET_TYPE: {NET_TYPE}")
    return net
