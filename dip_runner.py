"""dip_runner
=============

Trình chạy tối ưu Deep Image Prior (DIP) end-to-end.
- Hỗ trợ các mô hình trong models.get_net (skip/UNet/ResNet/dcgan).
- Toán tử quan sát A: blur/downsample/mask/compose.
- Regularization: TV và (tuỳ chọn) perceptual loss.
- Theo dõi: PSNR (quan sát) và PSNR_gt (nếu có GT), EMA và backtracking đơn giản.

Tích hợp cấu hình YAML:
- Có thể nạp file YAML (--config) để lấy tham số mặc định.
- Tham số CLI luôn ghi đè cấu hình YAML.
- Có thể in/lưu "cấu hình hiệu dụng" (sau hợp nhất) với --print_config/--dump_config.

Cách dùng (CLI ví dụ):
    python dip_runner.py --obs blurred.png --out deblurred.png --op blur --kernel_size 21 --kernel_sigma 2.0 --iters 3000
    python dip_runner.py --config config/config.yaml --print_config
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

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


# ====== Hỗ trợ cấu hình YAML và trích xuất cấu hình hiệu dụng ======
def _read_yaml(path: str) -> Dict[str, Any]:
    """Đọc file YAML an toàn. Nếu PyYAML không có, trả về {} và cảnh báo.

    Trả về:
        dict cấu hình (có thể lồng nhau) hoặc {} nếu lỗi.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[Cảnh báo] Không tìm thấy file cấu hình: {path}")
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            print("[Cảnh báo] Cấu trúc YAML không hợp lệ (mong đợi mapping)")
            return {}
        return data
    except Exception as e:  # pragma: no cover
        print(f"[Cảnh báo] Không thể đọc YAML ({e}); quay về cấu hình rỗng.")
        return {}


def _get_nested(d: Dict[str, Any], keys: str, default: Any = None) -> Any:
    """Lấy giá trị lồng nhau theo chuỗi khoá dạng 'a.b.c'."""
    cur = d
    for k in keys.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _build_run_kwargs_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Trích xuất tham số gọi run_dip từ cấu hình YAML (nếu có)."""
    kw: Dict[str, Any] = {}
    # IO
    kw["obspath"] = _get_nested(cfg, "io.obs")
    gt_path = _get_nested(cfg, "io.gt")
    kw["sharp_path"] = gt_path if gt_path else None
    kw["save_path"] = _get_nested(cfg, "io.out")
    # Model
    kw["net_type"] = _get_nested(cfg, "model.type")
    kw["input_depth"] = _get_nested(cfg, "model.input_depth")
    kw["upsample_mode"] = _get_nested(cfg, "model.upsample_mode")
    # Operator
    kw["op_name"] = _get_nested(cfg, "operator.name")
    kw["kernel_size"] = _get_nested(cfg, "operator.kernel_size")
    kw["kernel_sigma"] = _get_nested(cfg, "operator.kernel_sigma")
    kw["ds_factor"] = _get_nested(cfg, "operator.ds_factor")
    kw["ds_kernel"] = _get_nested(cfg, "operator.ds_kernel")
    # Optim
    kw["lr"] = _get_nested(cfg, "optim.lr")
    kw["num_iter"] = _get_nested(cfg, "optim.num_iter")
    kw["show_every"] = _get_nested(cfg, "optim.show_every")
    kw["ema_decay"] = _get_nested(cfg, "optim.ema")
    kw["reg_noise_std"] = _get_nested(cfg, "optim.reg_noise_std")
    # Loss
    kw["tv_weight"] = _get_nested(cfg, "loss.tv_weight")
    kw["use_percep"] = _get_nested(cfg, "loss.use_percep")
    kw["percep_weight"] = _get_nested(cfg, "loss.percep_weight")
    # Seed
    kw["seed"] = _get_nested(cfg, "seed")
    # Lọc None để không ghi đè mặc định của CLI khi thiếu trong YAML
    return {k: v for k, v in kw.items() if v is not None}


def _merge_cli_over_yaml(
    parser: argparse.ArgumentParser, args: argparse.Namespace, base: Dict[str, Any]
) -> Dict[str, Any]:
    """Hợp nhất tham số YAML (base) với CLI (args). CLI luôn có quyền ưu tiên nếu khác mặc định.

    Chiếu CLI -> tham số run_dip.
    """
    merged = dict(base)
    # Bản đồ tên tham số CLI -> tên tham số run_dip
    mapping = {
        "obs": "obspath",
        "gt": "sharp_path",
        "out": "save_path",
        "net": "net_type",
        "input_depth": "input_depth",
        "lr": "lr",
        "iters": "num_iter",
        "op": "op_name",
        "kernel_size": "kernel_size",
        "kernel_sigma": "kernel_sigma",
        "ds_factor": "ds_factor",
        "ds_kernel": "ds_kernel",
        "tv": "tv_weight",
        "percep": "use_percep",
        "percep_w": "percep_weight",
        "seed": "seed",
        "show_every": "show_every",
        "ema": "ema_decay",
        "reg_noise": "reg_noise_std",
        "upsample": "upsample_mode",
    }

    for cli_name, run_name in mapping.items():
        cli_val = getattr(args, cli_name, None)
        try:
            default_val = parser.get_default(cli_name)
        except Exception:
            default_val = None
        # Nếu người dùng truyền khác mặc định, coi như override
        if cli_val != default_val:
            merged[run_name] = cli_val if cli_name != "gt" else (cli_val or None)

    return merged


def _build_effective_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Tạo cấu hình hiệu dụng (phản ánh chính xác tham số run_dip sẽ dùng)."""
    eff = {
        "io": {
            "obs": kwargs.get("obspath"),
            "gt": kwargs.get("sharp_path"),
            "out": kwargs.get("save_path"),
        },
        "model": {
            "type": kwargs.get("net_type"),
            "input_depth": kwargs.get("input_depth"),
            "upsample_mode": kwargs.get("upsample_mode"),
        },
        "operator": {
            "name": kwargs.get("op_name"),
            "kernel_size": kwargs.get("kernel_size"),
            "kernel_sigma": kwargs.get("kernel_sigma"),
            "ds_factor": kwargs.get("ds_factor"),
            "ds_kernel": kwargs.get("ds_kernel"),
        },
        "optim": {
            "lr": kwargs.get("lr"),
            "num_iter": kwargs.get("num_iter"),
            "show_every": kwargs.get("show_every"),
            "ema": kwargs.get("ema_decay"),
            "reg_noise_std": kwargs.get("reg_noise_std"),
        },
        "loss": {
            "tv_weight": kwargs.get("tv_weight"),
            "use_percep": kwargs.get("use_percep"),
            "percep_weight": kwargs.get("percep_weight"),
        },
        "seed": kwargs.get("seed"),
    }
    return eff


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
    # Tuỳ chọn nạp và xuất cấu hình
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Đường dẫn file YAML cấu hình để nạp (tuỳ chọn)",
    )
    p.add_argument(
        "--print_config",
        action="store_true",
        help="In cấu hình hiệu dụng (sau khi hợp nhất YAML và CLI) rồi tiếp tục chạy",
    )
    p.add_argument(
        "--dump_config",
        type=str,
        default="",
        help="Lưu cấu hình hiệu dụng ra file (.yaml hoặc .json tuỳ đuôi)",
    )
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

    # 1) Đọc YAML (nếu có) và trích xuất tham số cho run_dip
    yaml_cfg = _read_yaml(args.config) if args.config else {}
    base_kwargs = _build_run_kwargs_from_cfg(yaml_cfg)
    # 2) Hợp nhất với CLI (CLI có quyền ưu tiên nếu khác mặc định)
    run_kwargs = _merge_cli_over_yaml(p, args, base_kwargs)

    # 3) Trích xuất và in/lưu cấu hình hiệu dụng theo yêu cầu
    eff_cfg = _build_effective_config(run_kwargs)
    if args.print_config:
        print("===== Cấu hình hiệu dụng (sau hợp nhất) =====")
        try:
            import yaml  # type: ignore

            print(yaml.safe_dump(eff_cfg, sort_keys=False, allow_unicode=True))
        except Exception:
            print(json.dumps(eff_cfg, indent=2, ensure_ascii=False))

    if args.dump_config:
        out_path = args.dump_config
        try:
            if out_path.lower().endswith((".yml", ".yaml")):
                import yaml  # type: ignore

                with open(out_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(eff_cfg, f, sort_keys=False, allow_unicode=True)
            else:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(eff_cfg, f, indent=2, ensure_ascii=False)
            print(f"Đã lưu cấu hình hiệu dụng: {out_path}")
        except Exception as e:
            print(f"[Cảnh báo] Không thể lưu cấu hình hiệu dụng: {e}")

    # 4) Chạy DIP với tham số đã hợp nhất
    run_dip(**run_kwargs)


if __name__ == "__main__":
    main()
