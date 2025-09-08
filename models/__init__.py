"""models.__init__
===================

Factory cung cấp hàm :func:`get_net` để khởi tạo các kiến trúc mạng dùng trong
thiết lập Deep Image Prior (DIP) cho bài toán khử mờ / khôi phục ảnh.

Deep Image Prior khai thác inductive bias của cấu trúc mạng CNN (không cần dữ liệu
huấn luyện ngoài) bằng cách tối ưu trực tiếp tham số mạng nhằm khớp ảnh suy hao.
Việc lựa chọn kiến trúc ảnh hưởng mạnh đến: tốc độ hội tụ, mức độ duy trì chi tiết
biên / texture và nguy cơ overfitting noise (early stopping quan trọng).

Các kiến trúc hỗ trợ:
1. 'skip'          : Encoder–decoder nhiều tầng có skip connections (tương tự U-Net
                      nhưng cấu hình linh hoạt số kênh từng nhánh). Thường cho chất
                      lượng tốt và phổ biến trong DIP gốc.
2. 'texture_nets'  : Mạng tổng hợp texture đa tỉ lệ (multi-scale texture synthesis).
                      Ít dùng cho deblur trực tiếp nhưng hữu ích khi cần priors dạng
                      pattern / lặp.
3. 'ResNet'        : Khối residual chồng nhiều block. Phù hợp khi muốn dòng gradient
                      ổn định và chiều sâu trung bình (medium-depth) thay vì dạng
                      encoder–decoder.
4. 'UNet'          : Biến thể chuẩn U-Net (có thể khác nhẹ ở tham số feature_scale,
                      upsample_mode). Dễ hiểu, cân bằng giữa chi tiết cục bộ & ngữ cảnh.
5. 'identity'      : Trả về ``nn.Sequential()`` rỗng; chỉ hợp lệ khi ``input_depth == n_channels``.
                      Dùng để kiểm tra baseline hoặc debug pipeline (không học biểu diễn).

Hướng dẫn chọn nhanh:
* Khử mờ (motion / defocus) tổng quát: 'skip' hoặc 'UNet'.
* Ảnh có nhiều biên sắc nét, cần giữ cấu trúc nhỏ: 'skip' với nhiều kênh skip_n11.
* Muốn mô hình gọn hoặc thử chiều sâu residual: 'ResNet'.
* Khảo sát pattern lặp hoặc texture tổng hợp: 'texture_nets'.
* Kiểm thử vòng tối ưu: 'identity'.

Tham số chính (tóm tắt):
------------------------
NET_TYPE : str
    Loại mạng như mô tả trên.
input_depth : int
    Số kênh của noise đầu vào (ví dụ 32, 64). Noise Gaussian / Uniform có shape (B, input_depth, H, W).
pad : str
    Kiểu padding cho convolution ("reflection", "zero", ...). Reflection thường giảm artefact biên.
upsample_mode : str
    Cách nội suy trong decoder ("bilinear" | "nearest"). Bilinear mượt, nearest giữ biên sắc nét hơn nhưng có thể blocky.
n_channels : int
    Số kênh đầu ra; ảnh RGB = 3, grayscale = 1.
act_fun : str
    Tên hàm kích hoạt nội bộ ("LeakyReLU", "ELU", ...). Ảnh hưởng smoothing / gradient flow.
skip_n33d / skip_n33u / skip_n11 : int | list[int]
    Cấu hình số kênh cho từng tầng encoder (down), decoder (up) và đường skip trong mạng 'skip'.
num_scales : int
    Số level encoder–decoder (độ sâu đa tỉ lệ) cho mạng 'skip'.
downsample_mode : str
    Cách giảm kích thước ("stride", "avg", "max"). "stride" = conv stride>1.
need_sigmoid : bool
    Thêm sigmoid cuối để ép output vào [0,1]. Nếu dùng chuẩn hóa khác (ví dụ training với range -1..1) đặt False.
res_blocks : int
    Số residual blocks cho 'ResNet'.
res_feat : int
    Số kênh đặc trưng trong mỗi residual block.
norm_layer : Callable
    Lớp chuẩn hóa dùng trong một số kiến trúc (BatchNorm2d, InstanceNorm2d, LayerNorm...).

Ghi chú triển khai:
-------------------
* Hàm tạo các list kênh nếu tham số đầu vào dạng int (broadcast ra chiều num_scales).
* Không thay đổi hành vi gốc; chỉ bổ sung chú thích phân tích.
* Nếu mở rộng, có thể thêm kiến trúc mới bằng cách bổ sung nhánh elif khác trong get_net.

"""

from .skip import skip
from .texture_nets import get_texture_nets
from .resnet import ResNet
from .unet import UNet

import torch.nn as nn


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
):
    """Trả về instance mạng theo ``NET_TYPE``.

    Lưu ý:
    * Không áp dụng weight init tùy biến ở đây (giả định kiến trúc con tự xử lý hoặc PyTorch mặc định).
    * Đối với DIP, thường khởi tạo noise cố định một lần và tối ưu trực tiếp tham số mạng.
    * ``need_sigmoid=False`` nếu bạn xử lý ảnh dạng float chuẩn hóa (ví dụ mean/std) và sẽ convert sau.

    Parameters
    ----------
    NET_TYPE : str
        Một trong {'skip','texture_nets','ResNet','UNet','identity'}.
    input_depth : int
        Số kênh của tensor đầu vào (thường là noise). Chọn lớn hơn (32–64) giúp biểu diễn phong phú hơn.
    pad : str
        Kiểu padding cho conv; 'reflection' giảm viền giả so với 'zero'.
    upsample_mode : str
        Kiểu nội suy khi up-sampling ('bilinear' mượt, 'nearest' sắc nhưng có thể răng cưa).
    n_channels : int
        Số kênh ảnh đầu ra mong muốn.
    act_fun : str
        Hàm kích hoạt (tên) được các module con tra cứu.
    skip_n33d / skip_n33u / skip_n11 : int | list[int]
        Nếu int -> broadcast ra danh sách độ dài num_scales; nếu list -> dùng trực tiếp.
    num_scales : int
        Chiều sâu đa tỉ lệ của kiến trúc 'skip'.
    downsample_mode : str
        Cách giảm kích thước trong 'skip'.
    need_sigmoid : bool
        Thêm lớp sigmoid ở cuối để giới hạn [0,1].
    res_blocks : int
        Số residual blocks (ResNet).
    res_feat : int
        Số kênh nội bộ (ResNet).
    norm_layer : nn.Module
        Lớp chuẩn hóa (dùng cho ResNet hoặc UNet tuỳ cài đặt).

    Returns
    -------
    torch.nn.Module
        Mạng khởi tạo tương ứng.
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
        )
    elif NET_TYPE == "identity":
        # Identity: dùng chủ yếu để debug pipeline hoặc làm baseline (không học gì).
        assert input_depth == n_channels, "identity: input_depth phải bằng n_channels"
        net = nn.Sequential()
    else:
        raise ValueError(f"Unknown NET_TYPE: {NET_TYPE}")
    return net
