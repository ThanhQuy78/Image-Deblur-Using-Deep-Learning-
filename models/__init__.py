"""models package
=================
Factory get_net khởi tạo mạng cho các thí nghiệm khôi phục ảnh / DIP.
Chọn NET_TYPE trong: 'skip','texture_nets','ResNet','UNet','dcgan','identity'.
Dùng: net = get_net('skip', input_depth=32, n_channels=3).
Các hàm tiện ích: count_params, model_summary, smoke_test.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .skip import skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet
from .dcgan import dcgan

__all__ = [
    "get_net",
    "skip",
    "get_texture_nets",
    "ResNet",
    "UNet",
    "dcgan",
    "count_params",
    "model_summary",
    "smoke_test",
]


# ----------------- Lightweight utils (tự cung cấp) -----------------


def count_params(model: nn.Module) -> int:
    """Đếm số tham số trainable."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size=(1, 3, 64, 64)) -> str:
    """Tạo chuỗi tóm tắt ngắn gọn (không phụ thuộc third-party)."""
    n_params = count_params(model)
    try:
        dummy = torch.randn(*input_size)
        with torch.no_grad():
            out = model(dummy)
        out_shape = tuple(out.shape) if isinstance(out, torch.Tensor) else type(out)
    except Exception as e:  # pragma: no cover
        out_shape = f"Forward error: {e}"
    lines = [
        "Model Summary",
        "--------------",
        f"Input size  : {input_size}",
        f"Output shape: {out_shape}",
        f"Trainable params: {n_params}",
        "--------------",
    ]
    return "\n".join(lines)


def smoke_test(model: nn.Module, input_size=(1, 3, 32, 32)) -> bool:
    """Kiểm tra forward nhanh để xác nhận không lỗi runtime."""
    model.eval()
    try:
        with torch.no_grad():
            _ = model(torch.randn(*input_size))
        return True
    except Exception:  # pragma: no cover
        return False


# ----------------- Factory -----------------


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
    # Chuẩn hóa pad
    if pad not in {"reflection", "zero", "replication"}:
        raise ValueError(f"Unsupported pad: {pad}")
    # Đồng bộ output_act với need_sigmoid nếu chưa cung cấp
    if need_sigmoid and output_act is None:
        output_act = "sigmoid"
    if output_act not in {None, "sigmoid", "tanh", "identity"}:
        raise ValueError(f"Unsupported output_act: {output_act}")

    if NET_TYPE == "ResNet":
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
        net = get_texture_nets(
            inp=input_depth,
            ratios=[32, 16, 8, 4, 2, 1],
            fill_noise=False,
            pad=pad,
            need_sigmoid=need_sigmoid,
            output_channels=n_channels,
            output_act=output_act,
        )
    elif NET_TYPE == "UNet":
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
        if dcgan_ups < 3:
            raise ValueError("dcgan: num_ups phải >= 3 để đảm bảo kích thước hợp lệ")
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
        if input_depth != n_channels:
            raise ValueError(
                f"identity: yêu cầu input_depth==n_channels (got {input_depth}!={n_channels})"
            )
        net = nn.Sequential()
    else:
        raise ValueError(f"Unknown NET_TYPE: {NET_TYPE}")
    return net
