"""Gói loss
Cung cấp:
- PerceptualLoss: tính content/style + TV (tùy chọn)
- tv_loss: hàm TV độc lập (wrapper)
- vgg_preprocess_*: hàm tiền xử lý tiện dùng khi debug/visualize
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
