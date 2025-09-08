"""loss.vgg_modified
====================

Định nghĩa lớp VGGModified: phiên bản VGG19 được chỉnh sửa:
- Thay ReLU bằng LeakyReLU (giảm dead neurons, giữ gradient).
- Dùng AvgPool2d thay vì MaxPool hoặc giữ nguyên index nhưng pool trung bình giúp mượt hơn cho nhiệm vụ tái tạo.
- Thêm Dropout2d trong classifier để regularize (nếu dùng đến phần fully-connected).

Mục tiêu: cung cấp backbone trích xuất đặc trưng (features) mềm mại hơn khi dùng cho perceptual loss trong các phương pháp khôi phục ảnh (deblur, denoise, SR).
"""

import torch.nn as nn


class VGGModified(nn.Module):
    """Bọc lại VGG19 với thay đổi Activation + Pooling.

    Tham số:
        vgg19_orig (nn.Module): mô hình VGG19 chuẩn từ torchvision.
        slope (float): hệ số negative slope cho LeakyReLU.

    Lưu ý: Chỉ giữ lại cấu trúc features & classifier cần thiết; không xử lý phần softmax.
    """

    def __init__(self, vgg19_orig, slope=0.01):
        super(VGGModified, self).__init__()
        self.features = nn.Sequential()

        # --- Block 1 ---
        self.features.add_module(str(0), vgg19_orig.features[0])  # conv1_1
        self.features.add_module(str(1), nn.LeakyReLU(slope, True))  # relu1_1 -> leaky
        self.features.add_module(str(2), vgg19_orig.features[2])  # conv1_2
        self.features.add_module(str(3), nn.LeakyReLU(slope, True))
        self.features.add_module(str(4), nn.AvgPool2d((2, 2), (2, 2)))  # pool1 (avg)

        # --- Block 2 ---
        self.features.add_module(str(5), vgg19_orig.features[5])
        self.features.add_module(str(6), nn.LeakyReLU(slope, True))
        self.features.add_module(str(7), vgg19_orig.features[7])
        self.features.add_module(str(8), nn.LeakyReLU(slope, True))
        self.features.add_module(str(9), nn.AvgPool2d((2, 2), (2, 2)))

        # --- Block 3 ---
        self.features.add_module(str(10), vgg19_orig.features[10])
        self.features.add_module(str(11), nn.LeakyReLU(slope, True))
        self.features.add_module(str(12), vgg19_orig.features[12])
        self.features.add_module(str(13), nn.LeakyReLU(slope, True))
        self.features.add_module(str(14), vgg19_orig.features[14])
        self.features.add_module(str(15), nn.LeakyReLU(slope, True))
        self.features.add_module(str(16), vgg19_orig.features[16])
        self.features.add_module(str(17), nn.LeakyReLU(slope, True))
        self.features.add_module(str(18), nn.AvgPool2d((2, 2), (2, 2)))

        # --- Block 4 ---
        self.features.add_module(str(19), vgg19_orig.features[19])
        self.features.add_module(str(20), nn.LeakyReLU(slope, True))
        self.features.add_module(str(21), vgg19_orig.features[21])
        self.features.add_module(str(22), nn.LeakyReLU(slope, True))
        self.features.add_module(str(23), vgg19_orig.features[23])
        self.features.add_module(str(24), nn.LeakyReLU(slope, True))
        self.features.add_module(str(25), vgg19_orig.features[25])
        self.features.add_module(str(26), nn.LeakyReLU(slope, True))
        self.features.add_module(str(27), nn.AvgPool2d((2, 2), (2, 2)))

        # --- Block 5 ---
        self.features.add_module(str(28), vgg19_orig.features[28])
        self.features.add_module(str(29), nn.LeakyReLU(slope, True))
        self.features.add_module(str(30), vgg19_orig.features[30])
        self.features.add_module(str(31), nn.LeakyReLU(slope, True))
        self.features.add_module(str(32), vgg19_orig.features[32])
        self.features.add_module(str(33), nn.LeakyReLU(slope, True))
        self.features.add_module(str(34), vgg19_orig.features[34])
        self.features.add_module(str(35), nn.LeakyReLU(slope, True))
        self.features.add_module(str(36), nn.AvgPool2d((2, 2), (2, 2)))

    def forward(self, x):
        return self.features(x)


__all__ = ["VGGModified"]
