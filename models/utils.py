import torch 
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def count_params(model: nn.Module) -> int:
    """Đếm số lượng tham số có thể huấn luyện trong mô hình."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model : nn.Module, input_size=(1,3,64,64)) -> None:
    """In ra tóm tắt kiến trúc mô hình."""
    n_params = count_params(model)
    dummy_input = torch.randn(*input_size)
    try:
        output = model(dummy_input)
        output_shape = output.shape
    except Exception as e:
        output_shape = f"Error during forward pass: {e}"
    line = []
    line.append(f"Model Summary:")
    line.append(f"----------------")
    line.append(f"Input size: {input_size}")
    line.append(f"Output size: {output_shape}")
    line.append(f"Number of parameters: {n_params}")
    line.append(f"----------------")
    return "\n".join(line)


def smoke_test(model: nn.Module, input_size=(1,3,64,64)) -> bool:
    model.eval()
    try:
        with torch.no_grad():
            dummy_input = torch.randn(*input_size)
            model(dummy_input)
        return True
    except Exception as e:
        print(f"Smoke test failed: {e}")
        return False
    
def load_img(path):
    """Load mô hình từ tệp."""
    return Image.open(path)

def save_torch_img(img_var, path):
    """Lưu tensor ảnh PyTorch thành tệp."""
    vutils.save_image(img_var.clamp(0,1), path)

def pil_to_np(img_pil):
    """Chuyển đổi ảnh PIL sang numpy array."""
    arr = np.array(img_pil)
    if len(arr.shape) == 3:
        arr = arr.transpose(2, 0, 1)
    else:
        arr = arr[None, :, :]  
    return arr.astype(np.float32) / 255.0

def np_to_pil(img_np):
    """Chuyển đổi numpy array sang ảnh PIL."""
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    if ar.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)
    return Image.fromarray(ar)

def np_to_torch(img_np):
    """Chuyển đổi numpy array sang tensor PyTorch."""
    return torch.from_numpy(img_np).unsqueeze(0)

def torch_to_np(img_var):
    """Chuyển đổi tensor PyTorch sang numpy array."""
    return img_var.detach().cpu().squeeze(0).numpy()

def get_noise(input_depth, method, spatial_size, noise_type="gaussian", var=1.0):
    """Sinh tensor noise.

    Parameters
    ----------
    input_depth : int        - số kênh
    method : 'noise'|'meshgrid'
    spatial_size : tuple     - (H, W)
    noise_type : 'gaussian'|'uniform'
    var : float              - độ lệch chuẩn Gaussian hoặc biên độ uniform

    Returns
    -------
    torch tensor 4D (1, input_depth, H, W)
    """
    if method == "noise":
        if noise_type == "gaussian":
            noise = torch.randn(1, input_depth, spatial_size[0], spatial_size[1]) * var
        elif noise_type == "uniform":
            noise = (torch.rand(1, input_depth, spatial_size[0], spatial_size[1]) - 0.5) * var
        else:
            assert False
    elif method == "meshgrid":
        assert input_depth == 2
        X, Y = np.meshgrid(
            np.linspace(0, 1, spatial_size[1]), np.linspace(0, 1, spatial_size[0])
        )
        meshgrid = np.concatenate([X[None, :, :], Y[None, :, :]], 0)
        noise = torch.from_numpy(meshgrid).float()[None, :, :, :]
    else:
        assert False
    return noise

def psnr(im1, im2):
    """Tính PSNR giữa hai ảnh numpy (giá trị trong [0, 1])."""
    mse = np.mean((im1 - im2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))

def optimize(optimizer_type, parameters, closure, lr, num_iter):
    """Hàm tối ưu hoá với closure."""
    if optimizer_type == "LBFGS":
        print("Starting optimization with LBFGS")
        optimizer = torch.optim.LBFGS(parameters, lr=lr, max_iter=num_iter)
        for i in range(num_iter):
            def lbfgs_closure():
                optimizer.zero_grad()
                loss = closure()
                loss.backward()
                return loss
            optimizer.step(lbfgs_closure)
    elif optimizer_type == "adam":
        print("Starting optimization with Adam")
        optimizer = torch.optim.Adam(parameters, lr=lr)
        for i in range(num_iter):
            optimizer.zero_grad()
            loss = closure()
            loss.backward()
            optimizer.step()
    else:
        assert False, "Unknown optimizer type"

    

    


