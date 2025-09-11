"""dip_runner
=============

Trình chạy tối ưu Deep Image Prior (DIP) end-to-end.
- Hỗ trợ các mô hình trong models.get_net (skip/UNet/ResNet/dcgan).
- Toán tử quan sát A: blur/downsample/mask/compose/motion.
- Regularization: TV và (tuỳ chọn) perceptual loss.
- Theo dõi: PSNR (quan sát) và PSNR_gt (nếu có GT), EMA và backtracking đơn giản.

Tích hợp cấu hình YAML:
- Có thể nạp file YAML (--config) để lấy tham số mặc định.
- Tham số CLI luôn ghi đè cấu hình YAML.
- Có thể in/lưu "cấu hình hiệu dụng" (sau hợp nhất) với --print_config/--dump_config.
"""

import argparse
import os
import json
import csv
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from models import get_net
from utils.image_io import load_img, crop_image, pil_to_np, np_to_torch, save_torch_img
from utils.metrics import compute_psnr, compute_ssim
from utils.torch_utils import get_noise, get_params, torch_to_np
from utils.operators import (
    gaussian_kernel,
    Blur,
    Mask,
    DownsampleOp,
    Compose,
    MotionBlur,
    PiecewiseBlur,
    motion_kernel,
)

# PyYAML (tuỳ chọn)
try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

# Perceptual loss (tuỳ chọn)
try:
    from loss.perceptual_loss import PerceptualLoss
except Exception:  # pragma: no cover
    PerceptualLoss = None  # type: ignore


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Tổng biến phân (L1) đơn giản theo H,W."""
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


# ====== Hỗ trợ cấu hình YAML và trích xuất cấu hình hiệu dụng ======


def _read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    if path.endswith((".yml", ".yaml")):
        if yaml is None:
            raise RuntimeError(
                "Thiếu PyYAML: không thể đọc YAML; cài 'pyyaml' hoặc dùng JSON"
            )
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def _get_nested(d: Dict[str, Any], keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_run_kwargs_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Trích xuất các tham số cần cho run_dip từ cấu hình YAML."""
    out: Dict[str, Any] = {}
    out["obspath"] = _get_nested(cfg, "io.obs")
    out["sharp_path"] = _get_nested(cfg, "io.gt", None)
    out["save_path"] = _get_nested(cfg, "io.out", "outputs/deblurred.png")

    out["net_type"] = _get_nested(cfg, "model.type", "skip")
    out["input_depth"] = int(_get_nested(cfg, "model.input_depth", 32))
    out["upsample_mode"] = _get_nested(cfg, "model.upsample_mode", "bilinear")

    out["op_name"] = _get_nested(cfg, "operator.name", "identity")
    out["kernel_size"] = int(_get_nested(cfg, "operator.kernel_size", 21))
    out["kernel_sigma"] = float(_get_nested(cfg, "operator.kernel_sigma", 2.0))
    out["ds_factor"] = int(_get_nested(cfg, "operator.ds_factor", 2))
    out["ds_kernel"] = _get_nested(cfg, "operator.ds_kernel", "lanczos2")
    out["motion_len"] = int(_get_nested(cfg, "operator.motion_len", 9))
    out["motion_angle"] = float(_get_nested(cfg, "operator.motion_angle", 0.0))
    # piecewise blur
    out["grid_nx"] = int(_get_nested(cfg, "operator.grid_nx", 3))
    out["grid_ny"] = int(_get_nested(cfg, "operator.grid_ny", 3))
    out["pw_mode"] = _get_nested(cfg, "operator.pw_mode", "gradient")
    out["angle_min"] = float(_get_nested(cfg, "operator.angle_min", -10.0))
    out["angle_max"] = float(_get_nested(cfg, "operator.angle_max", 10.0))
    out["len_min"] = int(_get_nested(cfg, "operator.len_min", 5))
    out["len_max"] = int(_get_nested(cfg, "operator.len_max", 15))
    out["blend"] = bool(_get_nested(cfg, "operator.blend", True))
    out["blend_ratio"] = float(_get_nested(cfg, "operator.blend_ratio", 0.15))

    out["lr"] = float(_get_nested(cfg, "optim.lr", 1e-3))
    out["num_iter"] = int(_get_nested(cfg, "optim.num_iter", 3000))
    out["show_every"] = int(_get_nested(cfg, "optim.show_every", 100))
    out["ema_decay"] = float(_get_nested(cfg, "optim.ema", 0.99))
    out["reg_noise_std"] = float(_get_nested(cfg, "optim.reg_noise_std", 1.0 / 30.0))

    out["tv_weight"] = float(_get_nested(cfg, "loss.tv_weight", 1e-5))
    out["use_percep"] = bool(_get_nested(cfg, "loss.use_percep", False))
    out["percep_weight"] = float(_get_nested(cfg, "loss.percep_weight", 0.0))

    out["seed"] = int(_get_nested(cfg, "seed", 1337))
    return out


def _merge_cli_over_yaml(
    parser: argparse.ArgumentParser, args: argparse.Namespace, base: Dict[str, Any]
) -> Dict[str, Any]:
    """Trộn tham số CLI (ưu tiên) lên cấu hình YAML đã đọc."""
    merged = dict(base)
    # Map tên arg CLI -> khoá run_dip
    mapping = {
        "obs": "obspath",
        "gt": "sharp_path",
        "out": "save_path",
        "net": "net_type",
        "input_depth": "input_depth",
        "upsample": "upsample_mode",
        "op": "op_name",
        "kernel_size": "kernel_size",
        "kernel_sigma": "kernel_sigma",
        "ds_factor": "ds_factor",
        "ds_kernel": "ds_kernel",
        "motion_len": "motion_len",
        "motion_angle": "motion_angle",
        "lr": "lr",
        "iters": "num_iter",
        "show_every": "show_every",
        "ema": "ema_decay",
        "reg_noise": "reg_noise_std",
        "tv": "tv_weight",
        "percep": "use_percep",
        "percep_w": "percep_weight",
        "seed": "seed",
        "amp": "amp",
        "tile_size": "tile_size",
        "tile_overlap": "tile_overlap",
        "holdout_p": "holdout_p",
        "early_patience": "early_patience",
        "grid_nx": "grid_nx",
        "grid_ny": "grid_ny",
        "pw_mode": "pw_mode",
        "angle_min": "angle_min",
        "angle_max": "angle_max",
        "len_min": "len_min",
        "len_max": "len_max",
        "blend": "blend",
        "blend_ratio": "blend_ratio",
    }
    defaults = {
        a.dest: a.default
        for g in parser._action_groups
        for a in g._group_actions
        if hasattr(a, "dest")
    }
    for cli_key, run_key in mapping.items():
        val = getattr(args, cli_key, None)
        if val is not None and val != defaults.get(cli_key):
            merged[run_key] = val
    return merged


def build_operator(
    op_name: str, channels: int, device: torch.device, args
) -> nn.Module:
    """Khởi tạo toán tử quan sát A theo tên."""
    if op_name == "identity":
        return nn.Identity()
    if op_name == "blur":
        k = gaussian_kernel(ks=args.kernel_size, sigma=args.kernel_sigma, device=device)
        return Blur(k, channels=channels).to(device)
    if op_name == "motion":
        return MotionBlur(
            length=getattr(args, "motion_len", 9),
            angle_deg=getattr(args, "motion_angle", 0.0),
            channels=channels,
        ).to(device)
    if op_name == "piecewise":
        ny = int(getattr(args, "grid_ny", 3))
        nx = int(getattr(args, "grid_nx", 3))
        mode = str(getattr(args, "pw_mode", "gradient"))
        angle_min = float(getattr(args, "angle_min", -10.0))
        angle_max = float(getattr(args, "angle_max", 10.0))
        len_min = int(getattr(args, "len_min", getattr(args, "motion_len", 9)))
        len_max = int(getattr(args, "len_max", getattr(args, "motion_len", 9)))
        blend = bool(getattr(args, "blend", True))
        blend_ratio = float(getattr(args, "blend_ratio", 0.15))

        kernels = []
        if mode == "fixed":
            ang = float(getattr(args, "motion_angle", 0.0))
            ln = int(getattr(args, "motion_len", 9))
            for j in range(ny):
                row = []
                for i in range(nx):
                    row.append(motion_kernel(length=ln, angle_deg=ang, device=device))
                kernels.append(row)
        elif mode == "gradient":
            for j in range(ny):
                row = []
                for i in range(nx):
                    t = i / max(1, nx - 1)
                    ang = angle_min * (1 - t) + angle_max * t
                    ln = int(getattr(args, "motion_len", 9))
                    row.append(motion_kernel(length=ln, angle_deg=ang, device=device))
                kernels.append(row)
        else:  # random
            g = torch.Generator(device=device)
            if hasattr(args, "seed"):
                g.manual_seed(int(getattr(args, "seed")))
            for j in range(ny):
                row = []
                for i in range(nx):
                    ang = float(
                        torch.empty(1, generator=g)
                        .uniform_(angle_min, angle_max)
                        .item()
                    )
                    ln = int(
                        torch.empty(1, generator=g)
                        .uniform_(len_min, len_max + 1)
                        .item()
                    )
                    row.append(motion_kernel(length=ln, angle_deg=ang, device=device))
                kernels.append(row)
        return PiecewiseBlur(
            kernels=kernels,
            grid_size=(ny, nx),
            channels=channels,
            padding=getattr(args, "padding", "replicate"),
            blend=blend,
            blend_ratio=blend_ratio,
        ).to(device)
    if op_name == "mask":
        mask = torch.ones(1, 1, args.height, args.width, device=device)
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


def _tiled_forward(
    net: nn.Module, z: torch.Tensor, out_ch: int, tile: int, overlap: int
) -> torch.Tensor:
    """Suy luận theo ô cho noise z (phù hợp với DIP) để tiết kiệm bộ nhớ."""
    B, C, H, W = z.shape
    assert B == 1
    step = tile - overlap
    out = torch.zeros((1, out_ch, H, W), device=z.device, dtype=z.dtype)
    weight = torch.zeros_like(out)
    for y in range(0, H, step):
        for x in range(0, W, step):
            y1 = min(y + tile, H)
            x1 = min(x + tile, W)
            oz = net(z[:, :, y:y1, x:x1])
            out[:, :, y:y1, x:x1] += oz
            weight[:, :, y:y1, x:x1] += 1.0
    return out / weight.clamp_min(1e-6)


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
    amp: bool = False,
    tile_size: Optional[int] = None,
    tile_overlap: int = 32,
    holdout_p: float = 0.0,
    early_patience: int = 0,
    motion_len: int = 9,
    motion_angle: float = 0.0,
    # piecewise blur params
    grid_nx: int = 3,
    grid_ny: int = 3,
    pw_mode: str = "gradient",
    angle_min: float = -10.0,
    angle_max: float = 10.0,
    len_min: int = 5,
    len_max: int = 15,
    blend: bool = True,
    blend_ratio: float = 0.15,
):
    # Seed & device
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

    class Args:  # adapter đơn giản
        pass

    args = Args()
    args.kernel_size = kernel_size
    args.kernel_sigma = kernel_sigma
    args.ds_factor = ds_factor
    args.ds_kernel = ds_kernel
    args.height = H
    args.width = W
    args.motion_len = motion_len
    args.motion_angle = motion_angle
    # piecewise
    args.grid_nx = grid_nx
    args.grid_ny = grid_ny
    args.pw_mode = pw_mode
    args.angle_min = angle_min
    args.angle_max = angle_max
    args.len_min = len_min
    args.len_max = len_max
    args.blend = blend
    args.blend_ratio = blend_ratio
    args.padding = "replicate"

    A = build_operator(op_name, channels=img_np.shape[0], device=device, args=args)

    # Hold-out mask cho early stopping (nếu bật)
    train_mask = None
    hold_mask = None
    if holdout_p and holdout_p > 0:
        mask = torch.rand(1, 1, H, W, device=device)
        hold_mask = (mask < holdout_p).float()
        train_mask = 1.0 - hold_mask

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=amp and torch.cuda.is_available())

    # EMA & backtracking
    out_ema = None
    last_net = None
    psnr_last = 0.0
    best_hold = -1.0
    patience = 0
    i = 0

    def closure():
        nonlocal i, out_ema, last_net, psnr_last, z, patience, best_hold

        if reg_noise_std > 0:
            z = z_saved + (noise_buf.normal_() * reg_noise_std)

        with torch.cuda.amp.autocast(enabled=bool(scaler._enabled)):
            out = net(z)
            pred = A(out)
            # Loss tổng
            if train_mask is not None:
                loss = mse(pred * train_mask, img_t * train_mask)
            else:
                loss = mse(pred, img_t)
            if tv_weight > 0:
                loss = loss + tv_weight * tv_loss(out)
            if percep is not None and percep_weight > 0:
                loss = loss + percep_weight * percep(out, img_t)

        if scaler._enabled:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # EMA
        if out_ema is None:
            out_ema = out.detach()
        else:
            out_ema = out_ema * ema_decay + out.detach() * (1 - ema_decay)

        # Metrics đơn giản
        out_np = torch_to_np(out.detach().clamp(0, 1))
        psnr_val = compute_psnr(out_np, img_np)

        # Early stopping bằng hold-out (không dùng GT)
        if hold_mask is not None and i % show_every == 0:
            out_hold_np = torch_to_np((out.detach() * hold_mask).clamp(0, 1))
            gt_hold_np = torch_to_np((img_t.detach() * hold_mask).clamp(0, 1))
            psnr_hold = compute_psnr(out_hold_np, gt_hold_np)
            if psnr_hold > best_hold + 1e-6:
                best_hold = psnr_hold
                patience = 0
            else:
                patience += 1

        if i % show_every == 0:
            msg = f"Vòng lặp {i:05d} | Loss {float(loss.item()):.6f} | PSNR {psnr_val:.2f}"
            if sharp_np is not None:
                psnr_gt = compute_psnr(out_np, sharp_np)
                ssim_gt = compute_ssim(out_np, sharp_np)
                msg += f" | PSNR_gt {psnr_gt:.2f} | SSIM_gt {ssim_gt:.3f}"
            if hold_mask is not None:
                msg += f" | Patience {patience}"
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
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(num_iter):
        opt.zero_grad(set_to_none=True)
        _ = closure()
        if scaler._enabled:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if hold_mask is not None and early_patience and patience >= early_patience:
            print("Dừng sớm theo hold-out.")
            break

    # Suy luận cuối (tiled nếu yêu cầu)
    with torch.no_grad():
        if tile_size and tile_size > 0:
            out = _tiled_forward(
                net, z, out_ch=img_np.shape[0], tile=tile_size, overlap=tile_overlap
            )
        else:
            out = net(z)
    out = out.clamp(0, 1)
    if out_ema is not None:
        out = out_ema.clamp(0, 1)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    save_torch_img(out, save_path)
    print(f"Đã lưu: {save_path}")


def _dump_effective_config(eff: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.endswith((".yml", ".yaml")) and yaml is not None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(eff, f, allow_unicode=True, sort_keys=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(eff, f, indent=2, ensure_ascii=False)


def _run_gopro_batch(args: argparse.Namespace):
    """Chạy hàng loạt trên cấu trúc GoPro: duyệt blur/* và ánh xạ sang sharp/*."""
    root = args.gopro_root
    out_csv = args.gopro_csv or os.path.join("outputs", "gopro_results.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    rows = []
    for dirpath, _, filenames in os.walk(root):
        if os.path.sep + "blur" + os.path.sep not in dirpath:
            continue
        for fn in filenames:
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            blur_path = os.path.join(dirpath, fn)
            sharp_path = blur_path.replace(
                os.path.sep + "blur" + os.path.sep, os.path.sep + "sharp" + os.path.sep
            )
            name = os.path.splitext(os.path.basename(fn))[0]
            save_path = os.path.join(args.out_dir or "outputs", f"{name}_deblur.png")
            run_dip(
                obspath=blur_path,
                sharp_path=sharp_path if os.path.isfile(sharp_path) else None,
                save_path=save_path,
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
                amp=args.amp,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                holdout_p=args.holdout_p,
                early_patience=args.early_patience,
                motion_len=args.motion_len,
                motion_angle=args.motion_angle,
                grid_nx=args.grid_nx,
                grid_ny=args.grid_ny,
                pw_mode=args.pw_mode,
                angle_min=args.angle_min,
                angle_max=args.angle_max,
                len_min=args.len_min,
                len_max=args.len_max,
                blend=args.blend,
                blend_ratio=args.blend_ratio,
            )
            rows.append({"image": blur_path, "output": save_path})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=rows[0].keys() if rows else ["image", "output"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Đã lưu CSV: {out_csv}")


def main():
    p = argparse.ArgumentParser(description="Trình chạy Deep Image Prior (DIP)")
    # I/O và mô hình
    p.add_argument("--obs", type=str, default="blurred.png", help="Ảnh quan sát")
    p.add_argument("--gt", type=str, default="", help="Ảnh ground-truth (tuỳ chọn)")
    p.add_argument(
        "--out", type=str, default="result.png", help="Đường dẫn lưu kết quả"
    )
    p.add_argument(
        "--out_dir", type=str, default="outputs", help="Thư mục lưu khi chạy batch"
    )
    p.add_argument(
        "--net",
        type=str,
        default="skip",
        choices=["skip", "UNet", "ResNet", "dcgan"],
        help="Loại mạng",
    )
    p.add_argument("--input_depth", type=int, default=32)
    p.add_argument("--upsample", type=str, default="bilinear")

    # Toán tử quan sát
    p.add_argument(
        "--op",
        type=str,
        default="identity",
        choices=[
            "identity",
            "blur",
            "motion",
            "piecewise",
            "downsample",
            "blur_downsample",
            "mask",
        ],
        help="Toán tử quan sát",
    )
    p.add_argument("--kernel_size", type=int, default=21)
    p.add_argument("--kernel_sigma", type=float, default=2.0)
    p.add_argument("--ds_factor", type=int, default=2)
    p.add_argument("--ds_kernel", type=str, default="lanczos2")
    p.add_argument("--motion_len", type=int, default=9, help="Độ dài motion blur")
    p.add_argument(
        "--motion_angle", type=float, default=0.0, help="Góc motion blur (độ)"
    )
    # Piecewise options
    p.add_argument("--grid_nx", type=int, default=3, help="Số ô theo trục X")
    p.add_argument("--grid_ny", type=int, default=3, help="Số ô theo trục Y")
    p.add_argument(
        "--pw_mode",
        type=str,
        default="gradient",
        choices=["fixed", "gradient", "random"],
        help="Chế độ sinh kernel cho từng ô",
    )
    p.add_argument("--angle_min", type=float, default=-10.0)
    p.add_argument("--angle_max", type=float, default=10.0)
    p.add_argument("--len_min", type=int, default=5)
    p.add_argument("--len_max", type=int, default=15)
    p.add_argument("--blend_ratio", type=float, default=0.15)
    p.add_argument("--blend", dest="blend", action="store_true", default=True)
    p.add_argument("--no_blend", dest="blend", action="store_false")

    # Tối ưu
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--tv", type=float, default=1e-5, help="Trọng số total-variation")
    p.add_argument("--percep", action="store_true", help="Bật perceptual loss")
    p.add_argument(
        "--percep_w", type=float, default=0.0, help="Trọng số perceptual loss"
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--show_every", type=int, default=100)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--reg_noise", type=float, default=1.0 / 30.0)
    p.add_argument("--amp", action="store_true", help="Bật AMP (mixed precision)")
    p.add_argument(
        "--tile_size", type=int, default=0, help="Kích thước ô suy luận (0: tắt tiled)"
    )
    p.add_argument("--tile_overlap", type=int, default=32, help="Chồng lấn giữa các ô")
    p.add_argument(
        "--holdout_p", type=float, default=0.0, help="Tỷ lệ pixel hold-out cho dừng sớm"
    )
    p.add_argument(
        "--early_patience",
        type=int,
        default=0,
        help="Số lần không cải thiện trước khi dừng",
    )

    # Cấu hình YAML
    p.add_argument("--config", type=str, default="", help="Đường dẫn YAML cấu hình")
    p.add_argument(
        "--print_config",
        action="store_true",
        help="In cấu hình hiệu dụng rồi tiếp tục chạy",
    )
    p.add_argument(
        "--dump_config",
        type=str,
        default="",
        help="Lưu cấu hình hiệu dụng ra YAML/JSON",
    )

    # Batch GoPro
    p.add_argument(
        "--gopro_root",
        type=str,
        default="",
        help="Thư mục gốc GoPro (nếu đặt sẽ chạy batch)",
    )
    p.add_argument("--gopro_csv", type=str, default="", help="CSV ghi kết quả batch")

    args = p.parse_args()

    # Chế độ batch GoPro
    if args.gopro_root:
        _run_gopro_batch(args)
        return

    # Hợp nhất YAML + CLI
    base = {}
    if args.config:
        cfg = _read_yaml(args.config)
        base = _build_run_kwargs_from_cfg(cfg)
    eff = _merge_cli_over_yaml(p, args, base)

    # In/lưu cấu hình hiệu dụng (nếu yêu cầu)
    if args.print_config:
        print(json.dumps(eff, indent=2, ensure_ascii=False))
    if args.dump_config:
        _dump_effective_config(eff, args.dump_config)

    run_dip(
        obspath=eff.get("obspath", args.obs),
        sharp_path=eff.get("sharp_path", args.gt if args.gt else None),
        save_path=eff.get("save_path", args.out),
        net_type=eff.get("net_type", args.net),
        input_depth=int(eff.get("input_depth", args.input_depth)),
        lr=float(eff.get("lr", args.lr)),
        num_iter=int(eff.get("num_iter", args.iters)),
        op_name=eff.get("op_name", args.op),
        kernel_size=int(eff.get("kernel_size", args.kernel_size)),
        kernel_sigma=float(eff.get("kernel_sigma", args.kernel_sigma)),
        ds_factor=int(eff.get("ds_factor", args.ds_factor)),
        ds_kernel=eff.get("ds_kernel", args.ds_kernel),
        tv_weight=float(eff.get("tv_weight", args.tv)),
        use_percep=bool(eff.get("use_percep", args.percep)),
        percep_weight=float(eff.get("percep_weight", args.percep_w)),
        seed=int(eff.get("seed", args.seed)),
        show_every=int(eff.get("show_every", args.show_every)),
        ema_decay=float(eff.get("ema_decay", args.ema)),
        reg_noise_std=float(eff.get("reg_noise_std", args.reg_noise)),
        upsample_mode=eff.get("upsample_mode", args.upsample),
        amp=bool(eff.get("amp", args.amp)),
        tile_size=int(eff.get("tile_size", args.tile_size)),
        tile_overlap=int(eff.get("tile_overlap", args.tile_overlap)),
        holdout_p=float(eff.get("holdout_p", args.holdout_p)),
        early_patience=int(eff.get("early_patience", args.early_patience)),
        motion_len=int(eff.get("motion_len", args.motion_len)),
        motion_angle=float(eff.get("motion_angle", args.motion_angle)),
        grid_nx=int(eff.get("grid_nx", args.grid_nx)),
        grid_ny=int(eff.get("grid_ny", args.grid_ny)),
        pw_mode=eff.get("pw_mode", args.pw_mode),
        angle_min=float(eff.get("angle_min", args.angle_min)),
        angle_max=float(eff.get("angle_max", args.angle_max)),
        len_min=int(eff.get("len_min", args.len_min)),
        len_max=int(eff.get("len_max", args.len_max)),
        blend=bool(eff.get("blend", args.blend)),
        blend_ratio=float(eff.get("blend_ratio", args.blend_ratio)),
    )


if __name__ == "__main__":
    main()
