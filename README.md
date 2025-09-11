# Image-Deblur (Deep Image Prior)

Khoá luận/đồ án minh hoạ khử mờ ảnh bằng Deep Image Prior (DIP). Repository cung cấp mô hình dựng ảnh, toán tử quan sát (A), trình chạy DIP end-to-end, cùng notebook hướng dẫn đầy đủ.

- Không cần dữ liệu huấn luyện: tối ưu trực tiếp tham số mạng để khớp A(G(z;θ)) với ảnh quan sát y.
- Hỗ trợ khử mờ Gaussian, giảm mẫu, inpainting,… thông qua các toán tử quan sát có thể ghép nối.

## Tính năng chính
- Mô hình (models): skip, UNet, ResNet, dcgan; factory get_net đồng bộ chữ ký tham số, hỗ trợ pad/upsample/need_bias.
- Toán tử quan sát (utils/operators.py): Gaussian Blur, Downsample (anti-alias), Mask, Compose.
- Trình chạy DIP (dip_runner.py):
  - MSE + TV + (tuỳ chọn) Perceptual Loss.
  - EMA đầu ra, backtracking đơn giản, ghi log PSNR.
  - CLI tiện dụng cho các kịch bản phổ biến.
- Notebook deblur_full.ipynb: quy trình toàn diện từ thiết lập, tối ưu, đánh giá, suy luận tới xuất mô hình và benchmark.
- Cấu hình YAML (config.yaml): tham số hoá IO, model, operator, optim, loss, eval, export.

## Cấu trúc thư mục
- models/
  - __init__.py: factory get_net
  - common.py, skip.py, unet.py, resnet.py, dcgan.py, texture_nets.py, downsampler.py
- utils/
  - operators.py (A), image_io.py, metrics.py, torch_utils.py, visualization.py
- dip_runner.py: trình chạy DIP qua CLI hoặc API
- deblur_full.ipynb: notebook tổng hợp (Việt hoá)
- config.yaml: cấu hình mẫu dự án

## Yêu cầu & cài đặt
- Python >= 3.8
- PyTorch phù hợp với hệ thống (CUDA/CPU)
- Thư viện phụ trợ: pillow, numpy, scipy (nếu SSIM), v.v.

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

# Có GT để theo dõi PSNR_gt
python dip_runner.py --obs blurred.png --gt sharp.png --out outputs/deblurred.png --op blur --iters 2000

# Giảm mẫu (SR x2 theo DIP)
python dip_runner.py --obs blurred.png --out outputs/restore.png --op downsample --ds_factor 2 --iters 2000
```

Các tuỳ chọn CLI chính:
- --net: skip | UNet | ResNet | dcgan
- --op: identity | blur | downsample | blur_downsample | mask
- --percep, --percep_w: bật và chọn trọng số perceptual loss (nếu module khả dụng)
- --tv: trọng số total-variation

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
- operator: blur/downsample/compose và tuỳ chọn kernel.
- optim: lr, num_iter, reg_noise_std, ema, show_every.
- loss: mse_weight, tv_weight, use_percep, percep_weight.
- eval, export: báo cáo và xuất mô hình.

## Mô hình hỗ trợ
- skip: encoder-decoder nhiều scale với skip connections, dễ tuỳ biến số kênh qua các scale.
- UNet: U-Net cấu hình được (feature_scale, more_layers, concat_x,...).
- ResNet: chuỗi residual blocks, bias/BN/activation thống nhất.
- dcgan: generator tối giản, có tuỳ chọn ConvTranspose2d hoặc Upsample+Conv.

## Toán tử quan sát (A)
- Blur: làm mờ Gaussian depthwise.
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
