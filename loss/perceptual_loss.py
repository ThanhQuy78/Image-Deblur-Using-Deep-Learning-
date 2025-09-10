"""Perceptual / Style loss module (ổn định, linh hoạt).

Chức năng chính:
- Trích đặc trưng VGG19 (features hoặc full) hoặc phiên bản đã chỉnh sửa (LeakyReLU + AvgPool).
- So khớp content (features) hoặc style (Gram) với tuỳ chọn normal hoá Gram.
- Khoảng cách đặc trưng: MSE hoặc L1.
- Total Variation (TV) loss tích hợp.
- Cache đặc trưng target khi ảnh GT cố định (cache_target=True) để tăng tốc.
- Tự động xử lý input ở dải [-1,1] hoặc [0,1].
- Chọn layer bằng index hoặc tên (reluX_Y theo VGG19 chuẩn).

Tham số quan trọng:
  backbone_type : 'vgg19_features' | 'vgg19_full' | 'vgg19_modified'
  match_mode    : 'features' | 'gram'
  feature_dist  : 'mse' | 'l1'
  gram_norm     : 'chwd' (chia C*H*W) | 'hw' (chia H*W)
  tv_weight     : hệ số nhân TV L1 (0 = tắt)
  cache_target  : True -> cache đặc trưng target giữa các forward (tiện DIP 1 ảnh)
  return_components : True -> trả (total, {perceptual, tv})
"""

from __future__ import annotations
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional

try:
    from torch.hub import download_url_to_file
except Exception:
    download_url_to_file = None

# Thêm import torchvision.models với fallback
try:
    import torchvision.models as tv_models
except Exception:  # pragma: no cover
    tv_models = None

from .vgg_modified import VGGModified

# ---------------- HẰNG SỐ ----------------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_CAFFE_MEAN_BGR = torch.tensor([103.939, 116.779, 123.680]).view(1, 3, 1, 1)
_ALLOWED_BACKBONES = {"vgg19_features", "vgg19_full", "vgg19_modified"}
_DEFAULT_LAYER_IDS = [1, 6, 11, 20, 29]
_LAYER_NAME_MAP_VGG19 = {
    1: "relu1_1",
    3: "relu1_2",
    6: "relu2_1",
    8: "relu2_2",
    11: "relu3_1",
    13: "relu3_2",
    15: "relu3_3",
    17: "relu3_4",
    20: "relu4_1",
    22: "relu4_2",
    24: "relu4_3",
    26: "relu4_4",
    29: "relu5_1",
    31: "relu5_2",
    33: "relu5_3",
    35: "relu5_4",
}

# Đường dẫn mặc định tới VGG19 Caffe weights
DEFAULT_VGG19_WEIGHTS_URL = (
    "https://box.skoltech.ru/index.php/s/HPcOFQTjXxbmp4X/download"
)

# ---------------- TIỀN XỬ LÝ ----------------


def _to_01(x: torch.Tensor) -> torch.Tensor:
    if x.min() < 0:
        return (x + 1.0) / 2.0
    return x


def vgg_preprocess_pytorch(x: torch.Tensor) -> torch.Tensor:
    return (x - _IMAGENET_MEAN.to(x.device, x.dtype)) / _IMAGENET_STD.to(
        x.device, x.dtype
    )


def vgg_preprocess_caffe(x: torch.Tensor) -> torch.Tensor:
    r, g, b = torch.chunk(x, 3, dim=1)
    bgr = torch.cat([b, g, r], 1)
    return bgr * 255.0 - _CAFFE_MEAN_BGR.to(x.device, x.dtype)


# ---------------- BACKBONE LOADERS ----------------

# Hỗ trợ load weights VGG19 từ file/URL bên ngoài


def _load_state_dict_into_vgg(model: nn.Module, weights_path: str) -> nn.Module:
    sd = torch.load(weights_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        warnings.warn(
            f"Thiếu key khi load VGG19 từ {weights_path}: {list(missing)[:10]} ..."
        )
    if unexpected:
        warnings.warn(f"Key không mong đợi khi load VGG19: {list(unexpected)[:10]} ...")
    return model


def _resolve_external_weights(vgg_weights: Optional[str]) -> Optional[str]:
    if not vgg_weights:
        return None
    if isinstance(vgg_weights, str) and vgg_weights.lower().startswith(
        ("http://", "https://")
    ):
        if download_url_to_file is None:
            warnings.warn(
                "Không thể tải URL do thiếu download_url_to_file. Vui lòng tải thủ công."
            )
            return None
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "weights")
        )
        os.makedirs(base_dir, exist_ok=True)
        dest = os.path.join(base_dir, "vgg19_external.pth")
        try:
            download_url_to_file(vgg_weights, dest, progress=True)
            return dest
        except Exception as e:
            warnings.warn(f"Tải weights thất bại: {e}")
            return None
    if os.path.isfile(vgg_weights):
        return vgg_weights
    warnings.warn(f"Không tìm thấy file weights: {vgg_weights}")
    return None


def _load_vgg19_features(weights_path: Optional[str] = None) -> nn.Sequential:
    # Yêu cầu torchvision
    if tv_models is None:
        raise ImportError(
            "torchvision.models is required to load VGG19. Please install torchvision."
        )
    vgg = None
    if weights_path:
        try:
            try:
                vgg = tv_models.vgg19(weights=None)
            except Exception:
                vgg = tv_models.vgg19(pretrained=False)
            _load_state_dict_into_vgg(vgg, weights_path)
        except Exception as e:
            warnings.warn(
                f"Load external VGG19 weights lỗi ({e}), dùng ImageNet mặc định."
            )
    if vgg is None:
        try:
            vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        except Exception:
            vgg = tv_models.vgg19(pretrained=True)
    feats = vgg.features.eval()
    for p in feats.parameters():
        p.requires_grad = False
    return feats


def _load_vgg19_full(weights_path: Optional[str] = None) -> nn.Sequential:
    # Yêu cầu torchvision
    if tv_models is None:
        raise ImportError(
            "torchvision.models is required to load VGG19. Please install torchvision."
        )
    vgg = None
    if weights_path:
        try:
            try:
                vgg = tv_models.vgg19(weights=None)
            except Exception:
                vgg = tv_models.vgg19(pretrained=False)
            _load_state_dict_into_vgg(vgg, weights_path)
        except Exception as e:
            warnings.warn(
                f"Load external VGG19 weights lỗi ({e}), dùng ImageNet mặc định."
            )
    if vgg is None:
        try:
            vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        except Exception:
            vgg = tv_models.vgg19(pretrained=True)
    seq = nn.Sequential()
    idx = 0
    for m in vgg.features:
        seq.add_module(str(idx), m)
        idx += 1
    seq.add_module(str(idx), nn.AdaptiveAvgPool2d((7, 7)))
    idx += 1
    seq.add_module(str(idx), nn.Flatten())
    idx += 1
    for m in vgg.classifier:
        seq.add_module(str(idx), m)
        idx += 1
    for p in seq.parameters():
        p.requires_grad = False
    return seq.eval()


def _load_vgg19_modified(
    slope=0.01, weights_path: Optional[str] = None
) -> nn.Sequential:
    # Yêu cầu torchvision
    if tv_models is None:
        raise ImportError(
            "torchvision.models is required to load VGG19. Please install torchvision."
        )
    base = None
    if weights_path:
        try:
            try:
                base = tv_models.vgg19(weights=None)
            except Exception:
                base = tv_models.vgg19(pretrained=False)
            _load_state_dict_into_vgg(base, weights_path)
        except Exception as e:
            warnings.warn(
                f"Load external VGG19 weights lỗi ({e}), dùng ImageNet mặc định."
            )
    if base is None:
        try:
            base = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        except Exception:
            base = tv_models.vgg19(pretrained=True)
    mod = VGGModified(base, slope=slope).eval()
    for p in mod.parameters():
        p.requires_grad = False
    return mod.features


# ---------------- TV LOSS ----------------


def _tv_l1(x: torch.Tensor) -> torch.Tensor:
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def tv_loss(x: torch.Tensor, mode: str = "l1") -> torch.Tensor:
    if mode == "l1":
        return _tv_l1(x)
    return x.new_tensor(0.0)


# ---------------- PERCEPTUAL LOSS ----------------
class PerceptualLoss(nn.Module):
    def __init__(
        self,
        layers=None,
        layer_weights=None,
        use_caffe=False,
        use_modified=False,  # deprecated; dùng backbone_type
        match_mode="features",  # 'features' | 'gram'
        reduction="mean",  # 'mean' | 'sum'
        add_tv=0.0,
        tv_weight=None,
        device=None,
        backbone_type="vgg19_features",
        modified_slope=0.01,
        enforce_same_size=False,
        tv_mode="l1",  # 'l1' | 'none'
        return_components=False,
        feature_dist="mse",  # 'mse' | 'l1'
        gram_norm="chwd",  # 'chwd' (C*H*W) | 'hw' (H*W)
        cache_target=False,
        vgg_weights: Optional[str] = None,  # path hoặc URL đến weights VGG19 bên ngoài
    ):
        super().__init__()
        # ------- Validate -------
        if use_modified and backbone_type != "vgg19_modified":
            warnings.warn(
                "use_modified đã lỗi thời. Dùng backbone_type='vgg19_modified'. Override tự động."
            )
            backbone_type = "vgg19_modified"
        if backbone_type not in _ALLOWED_BACKBONES:
            raise ValueError(f"backbone_type phải thuộc {_ALLOWED_BACKBONES}")
        if match_mode not in {"features", "gram"}:
            raise ValueError("match_mode phải 'features' hoặc 'gram'")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction phải 'mean' hoặc 'sum'")
        if tv_mode not in {"l1", "none"}:
            raise ValueError("tv_mode phải 'l1' hoặc 'none'")
        if feature_dist not in {"mse", "l1"}:
            raise ValueError("feature_dist phải 'mse' hoặc 'l1'")
        if gram_norm not in {"chwd", "hw"}:
            raise ValueError("gram_norm phải 'chwd' hoặc 'hw'")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.match_mode = match_mode
        self.reduction = reduction
        self.enforce_same_size = enforce_same_size
        self.tv_mode = tv_mode
        self.feature_dist = feature_dist
        self.gram_norm = gram_norm
        self.cache_target = cache_target

        # TV weight (ưu tiên tv_weight nếu có)
        if tv_weight is None:
            tv_weight = add_tv
        else:
            if add_tv not in (0, 0.0):
                warnings.warn("'add_tv' đã lỗi thời, dùng 'tv_weight'.")
        tv_weight = 0.0 if tv_weight is None else float(tv_weight)
        if tv_weight < 0:
            raise ValueError("tv_weight phải >= 0")
        self.tv_weight = tv_weight
        self.return_components = return_components

        # Nếu không cung cấp, dùng URL mặc định (Caffe) và bật Caffe preprocessing
        if vgg_weights is None:
            vgg_weights = DEFAULT_VGG19_WEIGHTS_URL
            if not use_caffe:
                warnings.warn(
                    "Dùng VGG19 weights mặc định (Caffe). Tự động bật Caffe preprocessing."
                )
                use_caffe = True

        # Giải quyết đường dẫn/URL weights tùy chỉnh nếu có
        resolved_vgg_w = _resolve_external_weights(vgg_weights)

        # ------- Backbone -------
        if backbone_type == "vgg19_features":
            self.features = _load_vgg19_features(weights_path=resolved_vgg_w)
        elif backbone_type == "vgg19_full":
            self.features = _load_vgg19_full(weights_path=resolved_vgg_w)
        else:
            self.features = _load_vgg19_modified(
                slope=modified_slope, weights_path=resolved_vgg_w
            )
        self.features = self.features.to(self.device)

        # ------- Layers & weights -------
        requested = list(layers) if layers is not None else _DEFAULT_LAYER_IDS
        self.layers = self._normalize_layers(requested)
        max_idx = len(self.features) - 1
        valid = [i for i in self.layers if i <= max_idx]
        dropped = sorted(set(self.layers) - set(valid))
        if dropped:
            warnings.warn(f"Loại bỏ layer vượt phạm vi backbone: {dropped}")
        if not valid:
            raise RuntimeError("Không có layer hợp lệ để tính loss.")
        self.layers = valid
        self.layer_weights = self._make_layer_weights(layer_weights)

        # ------- Preprocess fn -------
        self.preprocess_fn = (
            vgg_preprocess_caffe if use_caffe else vgg_preprocess_pytorch
        )

        # ------- Hook storage -------
        self._hook_outputs = {}
        self._hooks_registered = False
        self._register_hooks()

        # Cache target
        self._cached_target_feats = None
        self._cached_target_sig = None

    # -------- Helpers --------
    def _normalize_layers(self, specs):
        out = []
        for sp in specs:
            if isinstance(sp, int):
                if sp < 0:
                    raise ValueError("Layer index phải >=0")
                out.append(sp)
            elif isinstance(sp, str):
                idx = next(
                    (k for k, v in _LAYER_NAME_MAP_VGG19.items() if v == sp), None
                )
                if idx is None:
                    raise ValueError(f"Tên layer không hợp lệ: {sp}")
                out.append(idx)
            else:
                raise TypeError("Layer phải là int hoặc str")
        return sorted(set(out))

    def _make_layer_weights(self, w):
        if w is None:
            return {i: 1.0 for i in self.layers}
        if isinstance(w, (int, float)):
            val = float(w)
            return {i: val for i in self.layers}
        if isinstance(w, dict):
            return {i: float(w.get(i, 1.0)) for i in self.layers}
        raise TypeError("layer_weights sai kiểu: None | float | dict")

    def _register_hooks(self):
        if self._hooks_registered:
            return

        def make_hook(i):
            def hook(_m, _i, _o):
                self._hook_outputs[i] = _o

            return hook

        for i, m in enumerate(self.features):
            if i in self.layers:
                m.register_forward_hook(make_hook(i))
        self._hooks_registered = True

    @staticmethod
    def _expand_gray(x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = self._expand_gray(x)
        x = _to_01(x)
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            x = x.float()
        return self.preprocess_fn(x)

    def _gram(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        f = x.view(b, c, h * w)
        g = torch.bmm(f, f.transpose(1, 2))
        if self.gram_norm == "chwd":
            return g / (c * h * w)
        else:
            return g / (h * w)

    def _feat_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.feature_dist == "mse":
            if self.reduction == "mean":
                return F.mse_loss(a, b)
            return F.mse_loss(a, b, reduction="sum")
        else:
            if self.reduction == "mean":
                return F.l1_loss(a, b)
            return F.l1_loss(a, b, reduction="sum")

    def _target_signature(self, t: torch.Tensor):
        return (tuple(t.shape), str(t.dtype), float(t.sum().item()))

    # -------- Forward --------
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if pred.numel() == 0 or target.numel() == 0:
            raise ValueError("Tensor đầu vào rỗng")
        if pred.shape[0] != target.shape[0]:
            raise ValueError("Batch size pred và target khác nhau")
        if pred.shape[-2:] != target.shape[-2:]:
            if self.enforce_same_size:
                raise ValueError("Kích thước khác nhau và enforce_same_size=True")
            pred = F.interpolate(
                pred, size=target.shape[-2:], mode="bilinear", align_corners=False
            )

        device = self.device
        pred = pred.to(device)
        target = target.to(device)
        pred_p = self._prepare(pred)
        tgt_p = self._prepare(target)

        # ---- target features (cache optional) ----
        target_feats = None
        if self.cache_target and self._cached_target_feats is not None:
            sig = self._target_signature(target)
            if sig == self._cached_target_sig:
                target_feats = self._cached_target_feats
        if target_feats is None:
            self._hook_outputs.clear()
            with torch.no_grad():
                _ = self.features(tgt_p)
                target_feats = {
                    k: v.detach()
                    for k, v in self._hook_outputs.items()
                    if k in self.layers
                }
            if self.cache_target:
                self._cached_target_feats = target_feats
                self._cached_target_sig = self._target_signature(target)

        # ---- pred features ----
        self._hook_outputs.clear()
        _ = self.features(pred_p)
        pred_feats = {k: v for k, v in self._hook_outputs.items() if k in self.layers}

        perceptual_sum = pred.new_tensor(0.0)
        for lid in self.layers:
            if lid not in pred_feats or lid not in target_feats:
                continue
            pf, tf = pred_feats[lid], target_feats[lid]
            if self.match_mode == "gram":
                pf, tf = self._gram(pf), self._gram(tf)
            li = self._feat_distance(pf, tf)
            perceptual_sum = perceptual_sum + li * self.layer_weights[lid]

        total = perceptual_sum
        tv_comp = pred.new_tensor(0.0)
        if self.tv_weight > 0 and self.tv_mode == "l1":
            tv_comp = _tv_l1(pred) * self.tv_weight
            total = total + tv_comp

        if self.return_components:
            return total, {
                "perceptual": perceptual_sum.detach(),
                "tv": tv_comp.detach(),
            }
        return total


__all__ = [
    "PerceptualLoss",
    "tv_loss",
    "vgg_preprocess_pytorch",
    "vgg_preprocess_caffe",
]
