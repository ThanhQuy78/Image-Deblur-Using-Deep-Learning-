"""Perceptual loss module (tối ưu hoá, sạch lỗi, linh hoạt).

Cung cấp:
- PerceptualLoss: so sánh đặc trưng VGG19 (content) hoặc Gram (style).
- tv_loss nội bộ (tùy chọn) với 2 chế độ: 'l1' | 'none'.
- Tiền xử lý PyTorch / Caffe.

Tối ưu mới:
- Tham số backbone_type: 'vgg19_features' | 'vgg19_full' | 'vgg19_modified'.
- Tham số enforce_same_size: False => tự resize pred; True => báo lỗi nếu lệch kích thước.
- Tham số tv_mode: 'l1' hoặc 'none'.
- Lọc layer không hợp lệ + cảnh báo mềm.
- Hook an toàn (đăng ký 1 lần), không tạo rác mỗi forward.
- Kiểm tra batch rỗng, dtype không phải float -> ép sang float32.
- Tùy chọn layer string hoặc int; loại bỏ trùng lặp.
"""

from __future__ import annotations
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg_modified import VGGModified

# ========================= HẰNG SỐ / CẤU HÌNH =========================
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_CAFFE_MEAN_BGR = torch.tensor([103.939, 116.779, 123.680]).view(1, 3, 1, 1)
_ALLOWED_BACKBONES = {"vgg19_features", "vgg19_full", "vgg19_modified"}
_DEFAULT_LAYER_IDS = [1, 6, 11, 20, 29]
_LAYER_NAME_MAP_VGG19 = {  # name reverse lookup hỗ trợ string
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

# ========================= TIỀN XỬ LÝ =========================


def _to_01(x):
    if x.min() < 0:  # giả định dải [-1,1]
        return (x + 1.0) / 2.0
    return x


def vgg_preprocess_pytorch(x):
    return (x - _IMAGENET_MEAN.to(x.device, x.dtype)) / _IMAGENET_STD.to(
        x.device, x.dtype
    )


def vgg_preprocess_caffe(x):
    r, g, b = torch.chunk(x, 3, dim=1)
    bgr = torch.cat([b, g, r], 1)
    return bgr * 255.0 - _CAFFE_MEAN_BGR.to(x.device, x.dtype)


# ========================= BACKBONE LOADERS =========================


def _load_vgg19_features() -> nn.Sequential:
    try:
        from torchvision import models as tv_models
    except Exception as e:
        raise ImportError(
            "Thiếu torchvision. Cài đặt bằng: pip install torchvision"
        ) from e

    try:
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
    except Exception:
        vgg = tv_models.vgg19(pretrained=True)
    feats = vgg.features.eval()
    for p in feats.parameters():
        p.requires_grad = False
    return feats


def _load_vgg19_full() -> nn.Sequential:
    try:
        from torchvision import models as tv_models
    except Exception as e:
        raise ImportError(
            "Thiếu torchvision. Cài đặt bằng: pip install torchvision"
        ) from e

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


def _load_vgg19_modified() -> nn.Sequential:
    try:
        from torchvision import models as tv_models
    except Exception as e:
        raise ImportError(
            "Thiếu torchvision. Cài đặt bằng: pip install torchvision"
        ) from e

    try:
        base = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
    except Exception:
        base = tv_models.vgg19(pretrained=True)
    mod = VGGModified(base, slope=0.01).eval()
    for p in mod.parameters():
        p.requires_grad = False
    return mod.features


# ========================= TV LOSS =========================


def _tv_l1(x):
    # L1 trực tiếp (nhanh hơn gọi F.l1_loss nhiều lần)
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


# Thêm hàm public tv_loss (wrapper) để dùng bên ngoài / __init__
def tv_loss(x, mode="l1"):
    if mode == "l1":
        return _tv_l1(x)
    return x.new_tensor(0.0)


# ========================= PERCEPTUAL LOSS =========================
class PerceptualLoss(nn.Module):
    def __init__(
        self,
        layers=None,
        layer_weights=None,
        use_caffe=False,
        use_modified=False,
        match_mode="features",
        reduction="mean",
        add_tv=0.0,
        tv_weight=None,
        device=None,
        backbone_type="vgg19_features",
        enforce_same_size=False,
        tv_mode="l1",
        return_components=False,
        feature_dist="mse",  # NEW: 'mse' | 'l1' cho khoảng cách feature
        gram_norm="chwd",  # NEW: 'chwd' (chia C*H*W) | 'hw' (chia H*W)
        cache_target=False,  # NEW: cache đặc trưng target giữa các forward
    ):
        """Khởi tạo PerceptualLoss.
        Tham số chính:
          layers: danh sách layer (int|str) để lấy đặc trưng (None dùng mặc định).
          layer_weights: trọng số mỗi layer (scalar|dict|None).
          use_caffe: dùng chuẩn hoá kiểu Caffe (BGR*255 - mean) thay vì PyTorch.
          match_mode: 'features' (content) hoặc 'gram' (style).
          reduction: 'mean' hoặc 'sum' cho MSE từng layer.
          add_tv / tv_weight: hệ số TV (ưu tiên tv_weight).
          enforce_same_size: True -> báo lỗi nếu kích thước lệch thay vì resize.
          tv_mode: 'l1' hoặc 'none'.
          return_components: True -> trả (total, {perceptual, tv}).
        """
        super().__init__()
        if use_modified and backbone_type != "vgg19_modified":
            warnings.warn(
                "use_modified đã lỗi thời; dùng backbone_type='vgg19_modified'. Tự động override."
            )
            backbone_type = "vgg19_modified"
        if backbone_type not in _ALLOWED_BACKBONES:
            raise ValueError(f"backbone_type phải thuộc {_ALLOWED_BACKBONES}")
        if match_mode not in {"features", "gram"}:
            raise ValueError("match_mode phải là 'features' hoặc 'gram'")
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction phải là 'mean' hoặc 'sum'")
        if tv_mode not in {"l1", "none"}:
            raise ValueError("tv_mode phải là 'l1' hoặc 'none'")
        if feature_dist not in {"mse", "l1"}:
            raise ValueError("feature_dist phải thuộc {'mse','l1'}")
        if gram_norm not in {"chwd", "hw"}:
            raise ValueError("gram_norm phải thuộc {'chwd','hw'}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.match_mode = match_mode
        self.reduction = reduction
        self.add_tv = float(add_tv)
        self.enforce_same_size = enforce_same_size
        self.tv_mode = tv_mode
        self.feature_dist = feature_dist
        self.gram_norm = gram_norm
        self.cache_target = cache_target

        # Load backbone
        if backbone_type == "vgg19_features":
            self.features = _load_vgg19_features()
        elif backbone_type == "vgg19_full":
            self.features = _load_vgg19_full()
        else:
            self.features = _load_vgg19_modified()
        self.features = self.features.to(self.device)

        # Chuẩn hoá layer
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

        # Trọng số layer
        self.layer_weights = self._make_layer_weights(layer_weights)

        # Preprocess fn
        self.preprocess_fn = (
            vgg_preprocess_caffe if use_caffe else vgg_preprocess_pytorch
        )

        # Hook
        self._hook_outputs = {}
        self._hooks_registered = False
        self._register_hooks()

        # Tham số mới
        if tv_weight is None:
            tv_weight = add_tv
        else:
            if add_tv not in (0, 0.0):
                warnings.warn("'add_tv' đã lỗi thời; sử dụng 'tv_weight'.")
        if tv_weight is None:
            tv_weight = 0.0
        if tv_weight < 0:
            raise ValueError("tv_weight phải >= 0")
        self.tv_weight = float(tv_weight)
        self.return_components = return_components
        self.add_tv = self.tv_weight  # giữ tương thích tên thuộc tính cũ

    # ----------------- Helpers -----------------
    def _normalize_layers(self, layer_specs):
        out = []
        for spec in layer_specs:
            if isinstance(spec, int):
                if spec < 0:
                    raise ValueError("Layer index phải >=0")
                out.append(spec)
            elif isinstance(spec, str):
                idx = next(
                    (k for k, v in _LAYER_NAME_MAP_VGG19.items() if v == spec), None
                )
                if idx is None:
                    raise ValueError(f"Tên layer không hợp lệ: {spec}")
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
    def _expand_gray(x):
        return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x

    def _prepare(self, x):
        x = self._expand_gray(x)
        x = _to_01(x)
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            x = x.float()
        return self.preprocess_fn(x)

    def _gram(self, x):
        # cập nhật theo gram_norm
        b, c, h, w = x.shape
        f = x.view(b, c, h * w)
        g = torch.bmm(f, f.transpose(1, 2))
        if self.gram_norm == "chwd":
            return g / (c * h * w)
        else:  # 'hw'
            return g / (h * w)

    def _feat_distance(self, a, b):
        if self.feature_dist == "mse":
            if self.reduction == "mean":
                return F.mse_loss(a, b)
            return F.mse_loss(a, b, reduction="sum")
        else:  # 'l1'
            if self.reduction == "mean":
                return F.l1_loss(a, b)
            return F.l1_loss(a, b, reduction="sum")

    def _target_signature(self, t):
        # signature đơn giản: shape + dtype + tổng giá trị (float)
        return (tuple(t.shape), str(t.dtype), float(t.sum().item()))

    # ----------------- Forward -----------------
    def forward(self, pred, target):
        """Tính tổng loss.
        Trả về:
          total hoặc (total, dict) nếu return_components=True.
        Quy trình:
          1. Chuẩn hoá kích thước (nếu cho phép).
          2. Chuẩn hoá giá trị & tiền xử lý.
          3. Trích đặc trưng target (no grad), sau đó pred (có grad).
          4. Tính MSE (hoặc Gram + MSE) từng layer + trọng số.
          5. Cộng TV nếu bật.
        """
        if pred.numel() == 0 or target.numel() == 0:
            raise ValueError("Tensor đầu vào rỗng.")
        if pred.shape[0] != target.shape[0]:
            raise ValueError("Batch size pred và target khác nhau.")
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

        # ===== Target features (có thể cache) =====
        use_cache = self.cache_target
        target_feats = None
        if use_cache and self._cached_target_feats is not None:
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
            if use_cache:
                self._cached_target_feats = target_feats
                self._cached_target_sig = self._target_signature(target)

        # ===== Pred features =====
        self._hook_outputs.clear()
        _ = self.features(pred_p)
        pred_feats = {k: v for k, v in self._hook_outputs.items() if k in self.layers}

        total = pred.new_tensor(0.0)
        perceptual_sum = pred.new_tensor(0.0)
        for lid in self.layers:
            if lid not in target_feats or lid not in pred_feats:
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


# ========================= EXPORTS =========================
__all__ = [
    "PerceptualLoss",
    "tv_loss",
    "vgg_preprocess_pytorch",
    "vgg_preprocess_caffe",
]
