"""Gói loss
=================
Cung cấp các thành phần tính hàm mất mát phục hồi ảnh:
- PerceptualLoss: So khớp đặc trưng sâu giữa ảnh dự đoán và ảnh chuẩn (content) hoặc so khớp thống kê Gram (style).
- tv_loss: Regularizer Total Variation L1 giúp làm mượt, giảm nhiễu hạt/ checkerboard.
- vgg_preprocess_pytorch / vgg_preprocess_caffe: Hai kiểu chuẩn hoá ảnh đầu vào tương thích VGG (chuẩn ImageNet hoặc phong cách Caffe).
"""

from .perceptual_loss import (
    PerceptualLoss,
    tv_loss,
    vgg_preprocess_pytorch,
    vgg_preprocess_caffe,
)

__all__ = [
    "PerceptualLoss",
    "tv_loss",
    "vgg_preprocess_pytorch",
    "vgg_preprocess_caffe",
]
