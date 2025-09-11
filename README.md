# Image-Deblur (Deep Image Prior)

Khoá luận/đồ án minh hoạ khử mờ ảnh bằng Deep Image Prior (DIP). Repository cung cấp mô hình dựng ảnh, toán tử quan sát (A), trình chạy DIP end-to-end.

- Không cần dữ liệu huấn luyện: tối ưu trực tiếp tham số mạng để khớp A(G(z;θ)) với ảnh quan sát y.
- Hỗ trợ khử mờ Gaussian, giảm mẫu, inpainting,… thông qua các toán tử quan sát có thể ghép nối.

## Tính năng chính
- Mô hình (models): skip, UNet, ResNet, dcgan; factory get_net đồng bộ chữ ký tham số, hỗ trợ pad/upsample/need_bias.
- Toán tử quan sát (utils/operators.py): Gaussian Blur, MotionBlur (độ dài/góc), PiecewiseBlur (mờ không đồng nhất theo lưới), Downsample (anti-alias), Mask, Compose.
- Trình chạy DIP (dip_runner.py):
  - MSE + TV + (tuỳ chọn) Perceptual Loss.
  - EMA đầu ra, backtracking đơn giản, ghi log PSNR.
  - AMP (mixed precision), suy luận theo ô (tile) cho ảnh lớn, early stopping bằng hold‑out mask.
  - CLI tiện dụng, hỗ trợ cấu hình YAML; in/lưu "cấu hình hiệu dụng" (--print_config/--dump_config).
  - Chế độ chạy hàng loạt cho GoPro (--gopro_root) và xuất CSV.
- Cấu hình YAML (config/config.yaml): tham số hoá IO, model, operator (kể cả motion), optim, loss, eval, export.

## Cấu trúc thư mục
- models/
  - __init__.py: factory get_net
  - common.py, skip.py, unet.py, resnet.py, dcgan.py, texture_nets.py, downsampler.py
- utils/
  - operators.py (A), image_io.py, metrics.py, torch_utils.py, visualization.py
- dip_runner.py: trình chạy DIP qua CLI hoặc API (YAML, AMP, tile, hold‑out, GoPro batch)
- deblur_full.ipynb: notebook tổng hợp (Việt hoá)
- config/config.yaml: cấu hình mẫu dự án

## Yêu cầu & cài đặt
- Python >= 3.8
- PyTorch phù hợp với hệ thống (CUDA/CPU)
- Thư viện phụ trợ: pillow, numpy, scipy (nếu SSIM), v.v.
  - PyYAML (tuỳ chọn) để đọc/ghi YAML; nếu thiếu sẽ tự động in/lưu JSON.

Cài đặt nhanh (gợi ý):
```bash
# Tạo môi trường (tuỳ chọn)
python -m venv .venv && .venv/Scripts/activate  # Windows PowerShell

# Cài đặt thư viện tối thiểu
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # thay bằng CPU/CUDA phù hợp
pip install pillow numpy
```

## Sử dụng nhanh (CLI)
Đặt ảnh mờ ở cùng thư mục (blurred.png). Tuỳ chọn đặt ảnh GT (sharp.png) để đánh giá.
```bash
# Khử mờ Gaussian không mù (non-blind)
python dip_runner.py --obs blurred.png --out outputs/deblurred.png \
  --op blur --kernel_size 21 --kernel_sigma 2.0 --iters 3000 --net skip --input_depth 32 \
  --percep --percep_w 0.05 --tv 1e-5 --ema 0.99 --show_every 100

# Khử mờ chuyển động (motion blur) xấp xỉ
a=21; L=9; ANG=15
python dip_runner.py --obs blurred.png --out outputs/deblur_motion.png \
  --op motion --motion_len $L --motion_angle $ANG --iters 3000

# Giảm mẫu (SR x2 theo DIP)
python dip_runner.py --obs blurred.png --out outputs/restore.png --op downsample --ds_factor 2 --iters 2000
```

### Cấu hình YAML & hợp nhất với CLI
- Có thể chuẩn bị cấu hình tại `config/config.yaml`.
- Nạp YAML bằng `--config`; mọi tham số truyền qua CLI sẽ luôn ghi đè giá trị trong YAML.
- Có thể in/lưu "cấu hình hiệu dụng" (sau khi hợp nhất) để tái lập thí nghiệm.

Ví dụ:
```bash
# In cấu hình hiệu dụng rồi chạy
python dip_runner.py --config config/config.yaml --print_config --obs my_blur.png --iters 2000

# Lưu cấu hình hiệu dụng ra YAML
python dip_runner.py --config config/config.yaml --dump_config outputs/effective.yaml

# Nếu không cài PyYAML, chương trình sẽ tự động lưu ở định dạng JSON
python dip_runner.py --config config/config.yaml --dump_config outputs/effective.json
```

### GoPro batch
```bash
# Chạy batch trên bộ GOPRO_Large/test (sẽ dò blur/* và ánh xạ sang sharp/* nếu có)
# Gợi ý cấu hình cho blur không đồng nhất: piecewise với góc biến thiên theo trục X
python dip_runner.py --gopro_root D:/data/GOPRO_Large/test --out_dir outputs/gopro --iters 2000 \
  --op piecewise --grid_nx 3 --grid_ny 3 --pw_mode gradient --angle_min -12 --angle_max 12 \
  --motion_len 9 --amp --tile_size 512 --tile_overlap 64 --holdout_p 0.05 --early_patience 2
```

## Các tuỳ chọn CLI chính:
- --net: skip | UNet | ResNet | dcgan
- --op: identity | blur | motion | piecewise | downsample | blur_downsample | mask
  - piecewise params: --grid_nx/--grid_ny, --pw_mode [fixed|gradient|random], --angle_min/--angle_max, --len_min/--len_max, --blend/--no_blend, --blend_ratio
- --percep, --percep_w: bật và chọn trọng số perceptual loss (nếu module khả dụng)
- --tv: trọng số total-variation
- --amp: bật mixed precision; --tile_size/--tile_overlap: suy luận theo ô; --holdout_p/--early_patience: dừng sớm
- --config/--print_config/--dump_config: YAML và cấu hình hiệu dụng; --gopro_root/--gopro_csv: chạy batch và xuất CSV

## Notebook
Mở deblur_full.ipynb và chạy tuần tự các ô:
- Dọn sạch môi trường, cấu hình đường dẫn, kiểm tra GPU.
- Nạp ảnh, xây dựng mô hình và toán tử quan sát A.
- Chạy tối ưu DIP với EMA, backtracking, log PSNR.
- Lưu kết quả, đánh giá PSNR/SSIM (nếu có GT), benchmark độ trễ, xuất TorchScript.

## Cấu hình (config.yaml)
Các nhóm tham số chính:
- project, io: đường dẫn vào/ra, thư mục checkpoint/logs.
- model: type, input_depth, n_channels, upsample_mode, pad, tham số phụ (ResNet,...)
- operator: blur/motion/downsample/compose và tuỳ chọn kernel.
- optim: lr, num_iter, reg_noise_std, ema, show_every.
- loss: mse_weight, tv_weight, use_percep, percep_weight.
- eval, export: báo cáo và xuất mô hình.
  
Ghi chú:
- Khi dùng `--config`, các khoá liên quan trực tiếp tới runner sẽ được đọc và ánh xạ vào tham số run_dip.
- Tham số qua CLI (ví dụ `--iters`, `--op`, `--percep_w`...) luôn ghi đè cấu hình YAML.

## Mô hình hỗ trợ
- skip: encoder-decoder nhiều scale với skip connections, dễ tuỳ biến số kênh qua các scale.
- UNet: U-Net cấu hình được (feature_scale, more_layers, concat_x,...).
- ResNet: chuỗi residual blocks, bias/BN/activation thống nhất.
- dcgan: generator tối giản, có tuỳ chọn ConvTranspose2d hoặc Upsample+Conv.

## Toán tử quan sát (A)
- Blur: làm mờ Gaussian depthwise.
- MotionBlur: làm mờ theo hướng (độ dài/góc) cho chuyển động tuyến tính.
- DownsampleOp: giảm mẫu chống alias (Lanczos/Box/Gauss).
- Mask: inpainting (áp mask nhị phân/mềm).
- Compose: ghép nhiều A tuần tự.

## Ví dụ API (Python)
```python
from dip_runner import run_dip

run_dip(
    obspath="blurred.png",
    sharp_path="sharp.png",
    save_path="outputs/deblurred.png",
    net_type="skip",
    input_depth=32,
    op_name="blur",
    kernel_size=21,
    kernel_sigma=2.0,
    num_iter=2000,
    use_percep=True,
    percep_weight=0.05,
    amp=True,
    tile_size=512,
    tile_overlap=64,
)
```

## Mẹo và khắc phục sự cố
- Ảnh đầu vào nên chuẩn hoá về [0,1], RGB 3 kênh.
- Nếu PSNR dao động mạnh, tăng EMA (0.99→0.995) hoặc giảm reg_noise_std.
- Perceptual loss yêu cầu module loss/perceptual_loss.py; tắt bằng cách bỏ --percep nếu không có.
- Với ResNet/dcgan, chọn upsample_mode phù hợp để giảm checkerboard.

## Ghi chú
- Chỉ cần 1 ảnh mờ (blurred.png) để chạy; ảnh GT là tuỳ chọn để báo cáo chỉ số.
- Kết quả mặc định được lưu tại outputs/.

## Tài liệu tham khảo
- Ulyanov, Dmitry, Andrea Vedaldi, and Victor Lempitsky. "Deep image prior." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
