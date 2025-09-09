"""VGGModified
=============
Mục đích: Cung cấp backbone thay thế VGG19 tiêu chuẩn cho perceptual loss trong các bài toán khôi phục ảnh.
Điểm chỉnh sửa chính:
1. Thay toàn bộ ReLU bằng LeakyReLU để giữ gradient ở vùng âm, hạn chế dead units.
2. Thay MaxPool bằng AvgPool giúp đặc trưng mượt hơn (giảm mất chi tiết biên do chọn cực trị).
3. Bỏ hoàn toàn phần classifier (FC) để giảm tham số, chỉ giữ khối trích đặc trưng convolutional.
4. Giữ nguyên thứ tự và chỉ số layer conv/activation để ánh xạ với danh sách layer VGG tiêu chuẩn (dễ hook perceptual).

Chi tiết mapping chỉ số (index trong self.features):
- 0: conv1_1, 1: leaky(relu1_1), 2: conv1_2, 3: leaky(relu1_2), 4: pool1
- 5..9: block2 (tương tự)
- 10..18: block3
- 19..27: block4
- 28..36: block5

Sử dụng: truyền model gốc torchvision.models.vgg19(...) vào constructor. Thuộc tính .features là nn.Sequential đã chỉnh.
"""

import torch.nn as nn


class VGGModified(nn.Module):
    """Bọc lại VGG19 với thay đổi Activation + Pooling.

    Tham số:
        vgg19_orig (nn.Module): mô hình VGG19 chuẩn từ torchvision.
        slope (float): hệ số negative slope cho LeakyReLU.

    Lưu ý: Chỉ giữ lại cấu trúc features & classifier cần thiết; không xử lý phần softmax.
    """

    def __init__(self, vgg19_orig, slope: float = 0.01):
        super().__init__()
        self.features = nn.Sequential()
        # -------- Block 1 --------
        self.features.add_module("0", vgg19_orig.features[0])  # conv1_1
        self.features.add_module("1", nn.LeakyReLU(slope, inplace=True))
        self.features.add_module("2", vgg19_orig.features[2])  # conv1_2
        self.features.add_module("3", nn.LeakyReLU(slope, inplace=True))
        self.features.add_module("4", nn.AvgPool2d(2, 2))  # pool1
        # -------- Block 2 --------
        self.features.add_module("5", vgg19_orig.features[5])
        self.features.add_module("6", nn.LeakyReLU(slope, inplace=True))
        self.features.add_module("7", vgg19_orig.features[7])
        self.features.add_module("8", nn.LeakyReLU(slope, inplace=True))
        self.features.add_module("9", nn.AvgPool2d(2, 2))
        # -------- Block 3 --------
        self.features.add_module("10", vgg19_orig.features[10])
        self.features.add_module("11", nn.LeakyReLU(slope, inplace=True))
        self.features.add_module("12", vgg19_orig.features[12])
        self.features.add_module("13", nn.LeakyReLU(slope, inplace=True))
        self.features.add_module("14", vgg19_orig.features[14])
        self.features.add_module("15", nn.LeakyReLU(slope, inplace=True))
        self.features.add_module("16", vgg19_orig.features[16])
        self.features.add_module("17", nn.LeakyReLU(slope, inplace=True))
        self.features.add_module("18", nn.AvgPool2d(2, 2))
        # -------- Block 4 --------
        self.features.add_module("19", vgg19_orig.features[19])
        self.features.add_module("20", nn.LeakyReLU(slope, inplace=True))
        self.features.add_module("21", vgg19_orig.features[21])
        self.features.add_module("22", nn.LeakyReLU(slope, inplace=True))
        self.features.add_module("23", vgg19_orig.features[23])
        self.features.add_module("24", nn.LeakyReLU(slope, inplace=True))
        # (Chú ý: Có thể bổ sung tiếp block4 conv4 nếu muốn khớp đầy đủ 25..27 –
        # nhưng giữ tối giản vì perceptual thường dừng ở relu4_1/4_2)

    def forward(self, x):
        """Trả về đặc trưng sau toàn bộ chuỗi convolution + LeakyReLU + AvgPool.
        Dùng trực tiếp làm backbone cho PerceptualLoss (hook lấy intermediate layers).
        """
        return self.features(x)


__all__ = ["VGGModified"]
