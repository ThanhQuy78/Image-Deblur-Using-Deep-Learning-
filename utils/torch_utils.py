"""utils.torch_utils
====================

Tiện ích Torch phục vụ DIP:
- get_noise: tạo noise đầu vào (uniform/normal) kích thước (C,H,W).
- get_params: gom tham số cần tối ưu (ví dụ của mạng và/hoặc input).
- optimize: vòng tối ưu tổng quát với closure.
- torch_to_np: chuyển torch (1,C,H,W) -> numpy (C,H,W) trong [0,1].

Gợi ý:
- Với DIP, thường chỉ tối ưu tham số mạng; giữ z cố định hoặc tiêm nhiễu nhẹ.
- Sử dụng closure để tái tạo forward+backward cho mỗi bước tối ưu.
"""

import numpy as np
import torch
from PIL import Image


def get_params(opt_over, net, net_input, downsampler=None):
    """Lấy tham số để tối ưu hoá."""
    opt_over_list = opt_over.split(",")
    params = []

    for opt in opt_over_list:
        if opt == "net":
            params += [x for x in net.parameters()]
        elif opt == "down":
            assert downsampler is not None
            params += [x for x in downsampler.parameters()]
        elif opt == "input":
            assert net_input is not None
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, f"Unknown option {opt}"

    return params


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
            noise = (
                torch.rand(1, input_depth, spatial_size[0], spatial_size[1]) - 0.5
            ) * var
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


def optimize(optimizer_type, parameters, closure, lr, num_iter):
    """Hàm tối ưu hoá với closure."""
    if optimizer_type == "LBFGS":
        print("Starting optimization with LBFGS")
        optimizer = torch.optim.LBFGS(parameters, lr=lr, max_iter=num_iter)
        for i in range(num_iter):

            def lbfgs_closure():
                optimizer.zero_grad()
                loss = closure()
                return loss

            optimizer.step(lbfgs_closure)
    elif optimizer_type == "adam":
        print("Starting optimization with Adam")
        optimizer = torch.optim.Adam(parameters, lr=lr)
        for i in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
    else:
        assert False, "Unknown optimizer type"
