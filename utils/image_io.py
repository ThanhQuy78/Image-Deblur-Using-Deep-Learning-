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
    vutils.save_image(img_var.clamp(0,1), path)