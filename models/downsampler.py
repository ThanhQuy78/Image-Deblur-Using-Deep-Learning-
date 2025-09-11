"""models.downsampler
=====================

Bộ lọc giảm mẫu chống alias cho ảnh/feature map.
- Hỗ trợ các kernel: 'lanczos2' | 'lanczos3' | 'gauss' | 'box'.
- Đóng gói dưới dạng nn.Module (Conv depthwise) với stride=factor.
- Tuỳ chọn preserve_size dùng pad replication để giữ nguyên kích thước biên.

Sử dụng:
- Được gọi gián tiếp từ hàm conv(...) trong models.common khi downsample_mode là 'lanczos2'/'lanczos3'.
- Có thể sử dụng trực tiếp như 1 operator quan sát A trong các bài toán SR.

Cung cấp lớp Downsampler tạo phép downsample với kernel tùy chọn (lanczos2, lanczos3,
gauss, box). Dùng trong các module khác (ví dụ conv trong common.py) để kiểm soát
aliasing khi giảm kích thước thay vì stride trực tiếp.

Thiết kế:
- Tạo kernel 2D trước (numpy) qua get_kernel theo loại & tham số.
- Gói kernel vào Conv2d depthwise (số kênh in == out, không trộn kênh) stride=factor.
- Tùy chọn preserve_size: pad replication sao cho output giữ nguyên spatial size.

Tham số chính:
--------------
 n_planes     : số kênh (áp dụng kernel giống nhau từng kênh)
 factor       : hệ số downsample (stride)
 kernel_type  : 'lanczos2'|'lanczos3'|'gauss12'|'gauss1sq2'|'lanczos'|'gauss'|'box'
 phase        : 0 hoặc 0.5 (căn chỉnh kernel) – 0.5 dùng cho box / một số variant lanczos
 kernel_width : chiều rộng kernel (suy ra nếu None từ loại)
 support      : bán kính hỗ trợ với lanczos (2 hoặc 3)
 sigma        : độ lệch chuẩn với gauss
 preserve_size: giữ kích thước đầu ra bằng cách pad trước

Ghi chú:
- Lanczos: sinc windowed -> bảo toàn tần số cao tốt hơn so với box/gauss nhỏ.
- Gauss: làm mờ nhẹ chống alias nhưng mềm cạnh.
- Box: đơn giản nhất, thường kém về chất lượng.
"""

import numpy as np
import torch
import torch.nn as nn


class Downsampler(nn.Module):
    """Module downsample với kernel tùy biến (đã bổ sung register_buffer cho kernel)."""

    def __init__(
        self,
        n_planes,
        factor,
        kernel_type,
        phase=0,
        kernel_width=None,
        support=None,
        sigma=None,
        preserve_size=False,
    ):
        super(Downsampler, self).__init__()

        assert phase in [0, 0.5], "phase should be 0 or 0.5"

        # Chuẩn hoá alias tên kernel rút gọn sang dạng chung
        if kernel_type == "lanczos2":
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = "lanczos"
        elif kernel_type == "lanczos3":
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = "lanczos"
        elif kernel_type == "gauss12":
            kernel_width = 7
            sigma = 1 / 2
            kernel_type_ = "gauss"
        elif kernel_type == "gauss1sq2":
            kernel_width = 9
            sigma = 1.0 / np.sqrt(2)
            kernel_type_ = "gauss"
        elif kernel_type in ["lanczos", "gauss", "box"]:
            kernel_type_ = kernel_type
        else:
            assert False, "wrong name kernel"

        # Sinh kernel numpy -> torch và đăng ký buffer để auto move device / lưu state dict
        kernel_np = get_kernel(
            factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma
        )
        kernel_torch = torch.from_numpy(kernel_np).float()
        self.register_buffer("kernel", kernel_torch)

        # Dựng conv depthwise sử dụng shape kernel
        downsampler = nn.Conv2d(
            n_planes,
            n_planes,
            kernel_size=self.kernel.shape,
            stride=factor,
            padding=0,
            bias=True,
        )
        with torch.no_grad():
            downsampler.weight.zero_()
            if downsampler.bias is not None:
                downsampler.bias.zero_()
            for i in range(n_planes):
                downsampler.weight.data[i, i] = self.kernel
        self.downsampler_ = downsampler

        # Nếu cần giữ kích thước: tính pad replication tương ứng
        if preserve_size:
            if self.kernel.shape[0] % 2 == 1:
                pad = int((self.kernel.shape[0] - 1) / 2.0)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.0)
            self.padding = nn.ReplicationPad2d(pad)

        self.preserve_size = preserve_size

    def forward(self, input):
        # Pad nếu giữ kích thước
        x = self.padding(input) if self.preserve_size else input
        self.x = x  # lưu debug
        return self.downsampler_(x)


def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    """Sinh kernel lọc 2D theo loại.

    Parameters
    ----------
    factor : int        - hệ số downsample (ảnh hưởng chuẩn hoá toạ độ lanczos)
    kernel_type : str   - 'lanczos'|'gauss'|'box'
    phase : float       - 0 hoặc 0.5 (dịch tâm mẫu)
    kernel_width : int  - kích thước lưới kernel
    support : int       - bán kính lanczos (2 hoặc 3)
    sigma : float       - sigma gaussian

    Returns
    -------
    np.ndarray kernel chuẩn hoá (tổng = 1)

    Ghi chú:
    - Lanczos: kernel(i,j) = sinc(i)*sinc(i/support)*sinc(j)*sinc(j/support)
    - Gauss  : chuẩn hoá theo 2πσ^2
    - Box    : đều nhau 1/(w^2)
    """
    assert kernel_type in ["lanczos", "gauss", "box"]

    # Khởi tạo kích thước (phase=0.5 giảm 1 đơn vị mỗi chiều trừ box)
    if phase == 0.5 and kernel_type != "box":
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])

    if kernel_type == "box":
        assert phase == 0.5, "Box filter is always half-phased"
        kernel[:] = 1.0 / (kernel_width * kernel_width)
    elif kernel_type == "gauss":
        assert sigma, "sigma is not specified"
        assert phase != 0.5, "phase 1/2 for gauss not implemented"
        center = (kernel_width + 1.0) / 2.0
        sigma_sq = sigma * sigma
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center) / 2.0
                dj = (j - center) / 2.0
                g = np.exp(-(di * di + dj * dj) / (2 * sigma_sq))
                kernel[i - 1][j - 1] = g / (2.0 * np.pi * sigma_sq)
    elif kernel_type == "lanczos":
        assert support, "support is not specified"
        center = (kernel_width + 1) / 2.0
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor
                    dj = abs(j + 0.5 - center) / factor
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor
                val = 1
                if di != 0:
                    val *= support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val /= np.pi * np.pi * di * di
                if dj != 0:
                    val *= support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val /= np.pi * np.pi * dj * dj
                kernel[i - 1][j - 1] = val
    else:
        assert False, "wrong method name"

    kernel /= kernel.sum()
    return kernel
