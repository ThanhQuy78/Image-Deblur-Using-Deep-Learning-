"""utils.image_io
==================

Hàm I/O ảnh dùng trong ví dụ DIP:
- load_img: nạp ảnh từ đường dẫn (PIL.Image RGB).
- crop_image: cắt ảnh để tương thích kích thước (nếu cần) cho mạng/toán tử.
- pil_to_np: chuyển PIL -> numpy trong [0,1], shape (C,H,W).
- np_to_torch: chuyển numpy -> torch (1,C,H,W) trong [0,1].
- save_torch_img: lưu tensor torch (1,C,H,W) về file ảnh (PNG/JPEG).

Quy ước:
- Mọi tensor ảnh trong pipeline ở dạng float32, miền [0,1].
- Thứ tự kênh: (C,H,W) cho numpy và (1,C,H,W) cho torch.
"""

from PIL import Image
import torchvision.utils as vutils


def load_img(path):
    """Load mô hình từ tệp."""
    return Image.open(path)


def crop_image(img_pil, d=32):
    """Cắt ảnh PIL sao cho kích thước là bội số của d."""
    (w, h) = img_pil.size
    w2 = w - (w % d)
    h2 = h - (h % d)

    bbox = [int((w - w2) / 2), int((h - h2) / 2), int((w + w2) / 2), int((h + h2) / 2)]

    img_cropped = img_pil.crop(bbox)
    return img_cropped


def save_torch_img(img_var, path):
    """Lưu tensor ảnh PyTorch thành tệp."""
    vutils.save_image(img_var.clamp(0, 1), path)


# Gợi ý: nếu cần hỗ trợ ảnh RGBA/Grayscale, chuẩn hoá về RGB 3 kênh trước khi chuyển đổi.
