"""dip_runner
=============

Runner tối ưu Deep Image Prior (DIP) end-to-end.
- Hỗ trợ các mô hình trong models.get_net (skip/UNet/ResNet/dcgan).
- Toán tử quan sát A: blur/downsample/mask/compose.
- Regularization: TV và (tuỳ chọn) perceptual loss.
- Theo dõi: PSNR (quan sát) và PSNR_gt (nếu có GT), EMA và backtracking đơn giản.

Cách dùng (CLI ví dụ):
  python dip_runner.py --obs blurred.png --out deblurred.png --op blur --kernel_size 21 --kernel_sigma 2.0 --iters 3000
"""

import argparse
import os
import torch
import torch.nn as nn
from typing import Optional

from models import get_net
from utils.image_io import load_img, crop_image, pil_to_np, np_to_torch, save_torch_img
from utils.metrics import compute_psnr
from utils.torch_utils import get_noise, get_params, optimize, torch_to_np
from utils.operators import gaussian_kernel, Blur, Mask, DownsampleOp, Compose

try:
    from loss.perceptual_loss import PerceptualLoss
except Exception:
    PerceptualLoss = None  # type: ignore


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    # Tổng biến phân (Total Variation) theo chuẩn L1 (trung bình) theo chiều dọc/ngang
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def build_operator(
    op_name: str, channels: int, device: torch.device, args
) -> nn.Module:
    """Khởi tạo toán tử quan sát A theo tên.

    Hỗ trợ:
    - identity: đồng nhất
    - blur: làm mờ Gaussian cố định
    - mask: inpainting với mask=1 giữ nguyên/0 che
    - downsample: giảm mẫu chống alias
    - blur_downsample: làm mờ rồi giảm mẫu
    """
    if op_name == "identity":
        return nn.Identity()
    if op_name == "blur":
        k = gaussian_kernel(ks=args.kernel_size, sigma=args.kernel_sigma, device=device)
        return Blur(k, channels=channels).to(device)
    if op_name == "mask":
        mask = torch.ones(1, 1, args.height, args.width, device=device)
        # Có thể mở rộng: nạp mask từ ảnh ngoài
        return Mask(mask).to(device)
    if op_name == "downsample":
        return DownsampleOp(factor=args.ds_factor, kernel_type=args.ds_kernel).to(
            device
        )
    if op_name == "blur_downsample":
        k = gaussian_kernel(ks=args.kernel_size, sigma=args.kernel_sigma, device=device)
        return Compose(
            [
                Blur(k, channels=channels),
                DownsampleOp(factor=args.ds_factor, kernel_type=args.ds_kernel),
            ]
        ).to(device)
    raise ValueError(f"Toán tử không hỗ trợ: {op_name}")


def run_dip(
    obspath: str,
    sharp_path: Optional[str],
    save_path: str,
    net_type: str = "skip",
    input_depth: int = 32,
    lr: float = 1e-3,
    num_iter: int = 3000,
    op_name: str = "identity",
    kernel_size: int = 21,
    kernel_sigma: float = 2.0,
    ds_factor: int = 2,
    ds_kernel: str = "lanczos2",
    tv_weight: float = 1e-5,
    use_percep: bool = False,
    percep_weight: float = 0.0,
    seed: int = 1337,
    show_every: int = 100,
    ema_decay: float = 0.99,
    reg_noise_std: float = 1.0 / 30.0,
    upsample_mode: str = "bilinear",
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Nạp ảnh quan sát y và (tuỳ chọn) GT
    img_pil = load_img(obspath)
    img_pil = crop_image(img_pil)
    img_np = pil_to_np(img_pil)
    img_t = np_to_torch(img_np).to(device)

    sharp_np = None
    if sharp_path and os.path.exists(sharp_path):
        sharp_pil = crop_image(load_img(sharp_path))
        sharp_np = pil_to_np(sharp_pil)

    # Khởi tạo mạng và noise z
    net = get_net(
        net_type,
        input_depth=input_depth,
        n_channels=img_np.shape[0],
        upsample_mode=upsample_mode,
    ).to(device)
    z = get_noise(input_depth, "noise", img_np.shape[1:], noise_type="uniform").to(
        device
    )
    z_saved = z.detach().clone()
    noise_buf = z.detach().clone()

    # MSE + (tuỳ chọn) perceptual loss
    mse = nn.MSELoss().to(device)
    percep = None
    if use_percep:
        if PerceptualLoss is None:
            raise RuntimeError(
                "PerceptualLoss không khả dụng; cần loss/perceptual_loss.py"
            )
        percep = PerceptualLoss(
            backbone_type="vgg19_modified",
            match_mode="features",
            feature_dist="l1",
            tv_weight=0.0,
            cache_target=True,
        ).to(device)

    # Toán tử quan sát A
    H, W = img_np.shape[1:]

    class Args:  # minimal adapter
        pass

    args = Args()
    args.kernel_size = kernel_size
    args.kernel_sigma = kernel_sigma
    args.ds_factor = ds_factor
    args.ds_kernel = ds_kernel
    args.height = H
    args.width = W

    A = build_operator(op_name, channels=img_np.shape[0], device=device, args=args)

    # EMA và sổ theo dõi
    out_ema = None
    last_net = None
    psnr_last = 0.0
    i = 0

    def closure():
        nonlocal i, out_ema, last_net, psnr_last, z

        # Tiêm nhiễu vào đầu vào để regularize
        if reg_noise_std > 0:
            z = z_saved + (noise_buf.normal_() * reg_noise_std)

        out = net(z)
        pred = A(out)

        # EMA để làm mượt quan sát
        if out_ema is None:
            out_ema = out.detach()
        else:
            out_ema = out_ema * ema_decay + out.detach() * (1 - ema_decay)

        # Loss tổng
        loss = mse(pred, img_t)
        if tv_weight > 0:
            loss = loss + tv_weight * tv_loss(out)
        if percep is not None and percep_weight > 0:
            loss = loss + percep_weight * percep(out, img_t)

        loss.backward()

        # Metrics đơn giản
        out_np = torch_to_np(out.detach().clamp(0, 1))
        psnr_val = compute_psnr(out_np, img_np)

        if sharp_np is not None:
            psnr_gt = compute_psnr(out_np, sharp_np)
        else:
            psnr_gt = None

        if i % show_every == 0:
            msg = f"Vòng lặp {i:05d} | Loss {float(loss.item()):.6f} | PSNR {psnr_val:.2f}"
            if psnr_gt is not None:
                msg += f" | PSNR_gt {psnr_gt:.2f}"
            print(msg)

        # Backtracking nếu PSNR quan sát giảm mạnh
        if i % show_every == 0:
            if psnr_val - psnr_last < -5 and last_net is not None:
                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.data.copy_(new_param.to(device))
                return loss * 0
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psnr_last = psnr_val

        i += 1
        return loss

    params = get_params("net", net, z)
    optimize("adam", params, closure, lr=lr, num_iter=num_iter)

    out = net(z).detach()
    out = out.clamp(0, 1)
    if out_ema is not None:
        out = out_ema.clamp(0, 1)

    save_torch_img(out, save_path)
    print(f"Đã lưu: {save_path}")


def main():
    p = argparse.ArgumentParser(description="Trình chạy Deep Image Prior (DIP)")
    p.add_argument(
        "--obs", type=str, default="blurred.png", help="Đường dẫn ảnh quan sát"
    )
    p.add_argument("--gt", type=str, default="", help="Đường dẫn ảnh GT (tuỳ chọn)")
    p.add_argument(
        "--out", type=str, default="result.png", help="Đường dẫn lưu ảnh kết quả"
    )
    p.add_argument(
        "--net",
        type=str,
        default="skip",
        choices=["skip", "UNet", "ResNet", "dcgan"],
        help="Loại mạng sử dụng",
    )
    p.add_argument("--input_depth", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument(
        "--op",
        type=str,
        default="identity",
        choices=["identity", "blur", "downsample", "blur_downsample", "mask"],
        help="Toán tử quan sát",
    )
    p.add_argument("--kernel_size", type=int, default=21)
    p.add_argument("--kernel_sigma", type=float, default=2.0)
    p.add_argument("--ds_factor", type=int, default=2)
    p.add_argument("--ds_kernel", type=str, default="lanczos2")
    p.add_argument("--tv", type=float, default=1e-5, help="Trọng số total-variation")
    p.add_argument("--percep", action="store_true", help="Bật perceptual loss")
    p.add_argument(
        "--percep_w", type=float, default=0.0, help="Trọng số perceptual loss"
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--show_every", type=int, default=100)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--reg_noise", type=float, default=1.0 / 30.0)
    p.add_argument("--upsample", type=str, default="bilinear")

    args = p.parse_args()
    run_dip(
        obspath=args.obs,
        sharp_path=args.gt if args.gt else None,
        save_path=args.out,
        net_type=args.net,
        input_depth=args.input_depth,
        lr=args.lr,
        num_iter=args.iters,
        op_name=args.op,
        kernel_size=args.kernel_size,
        kernel_sigma=args.kernel_sigma,
        ds_factor=args.ds_factor,
        ds_kernel=args.ds_kernel,
        tv_weight=args.tv,
        use_percep=args.percep,
        percep_weight=args.percep_w,
        seed=args.seed,
        show_every=args.show_every,
        ema_decay=args.ema,
        reg_noise_std=args.reg_noise,
        upsample_mode=args.upsample,
    )


if __name__ == "__main__":
    main()
