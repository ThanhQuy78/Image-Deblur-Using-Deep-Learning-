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

import numpy as np
from PIL import Image
import torch


def load_img(path: str) -> Image.Image:
    """Nạp ảnh từ đường dẫn và chuẩn hoá về RGB."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def crop_image(img: Image.Image, mod: int = 32) -> Image.Image:
    """Cắt ảnh để H,W chia hết cho mod (phù hợp mạng encoder-decoder)."""
    w, h = img.size
    w2 = (w // mod) * mod
    h2 = (h // mod) * mod
    if w2 == w and h2 == h:
        return img
    return img.crop((0, 0, w2, h2))


def pil_to_np(img: Image.Image) -> np.ndarray:
    """PIL -> numpy float32 trong [0,1], shape (C,H,W)."""
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    # (H,W,C) -> (C,H,W)
    arr = arr.transpose(2, 0, 1)
    return arr


def np_to_torch(arr: np.ndarray) -> torch.Tensor:
    """numpy (C,H,W) [0,1] -> torch (1,C,H,W) float32."""
    if arr.ndim == 3:
        t = torch.from_numpy(arr)[None, ...]
    elif arr.ndim == 4 and arr.shape[0] == 1:
        t = torch.from_numpy(arr)
    else:
        raise ValueError("np_to_torch: kỳ vọng (C,H,W) hoặc (1,C,H,W)")
    return t.float()


def save_torch_img(t: torch.Tensor, path: str) -> None:
    """Lưu tensor (1,C,H,W) [0,1] về file ảnh."""
    x = t.detach().cpu().clamp(0, 1)
    if x.dim() == 4 and x.shape[0] == 1:
        x = x[0]
    if x.dim() != 3:
        raise ValueError("save_torch_img: kỳ vọng tensor (1,C,H,W) hoặc (C,H,W)")
    # (C,H,W) -> (H,W,C)
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(x).save(path)
