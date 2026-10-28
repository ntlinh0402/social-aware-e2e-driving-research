# STATUS

## LAPTOP
- Trạng thái máy: ASUS TUF A17 FA706II, Windows 11 Home (build 26200), **1 ổ Kingston NVMe SSD 512 GB** (C:42 GB / O:108 GB / F:115 GB trống — **không có HDD** như CLAUDE.md giả định), RAM 16 GB (tốt hơn 8 GB brief đoán), GPU **NVIDIA GTX 1650 Ti 4 GB** (driver 32.0.15.9200) + iGPU AMD Radeon. WSL2 đã bật, chưa cài distro. Repo clone sẵn ở `O:\research`. — cập nhật lần cuối: 2026-10-28
- Đang làm: A3 — chuẩn bị cài **WSL2 Ubuntu-22.04** làm env Linux cho stack ddv2 (quyết định: không boot dual-boot để giữ Windows làm việc song song; CUDA passthrough qua WSL2 với driver mới ổn với GTX 1650 Ti).
- Đã xong: xác định phần cứng; chọn hướng OS (WSL2 Ubuntu-22.04); chốt layout đường dẫn: `~/navsim_workspace/research` → symlink tới `/mnt/o/research`, `dataset` → `/mnt/f/navsim_dataset`, `DiffusionDriveV2` + `exp` nằm trong WSL ext4.
- Handoff mới nhất: (chưa có)
- Vướng mắc: chưa có; chờ user chạy `wsl --install -d Ubuntu-22.04` trong PowerShell admin, sau đó tiếp tục A3 (swap, conda, env ddv2, checkpoint).

## MAC
- Trạng thái máy: Darwin arm64, Apple M5, RAM 16 GB, ổ trống ~800 GiB; Miniforge ở ~/miniforge3 (chưa `conda init`, gọi bằng đường dẫn đầy đủ); chưa có tmux — cập nhật lần cuối: 2026-10-28
- Đang làm: A1 — ConstantVelocity mini: non_reactive đã chạy xong 3615/3615 (log logs/A1_cv_non_reactive_20261028-1249.log, chưa đọc điểm); reactive bị gián đoạn khi máy gập (~111/3615), cần chạy lại
- Đã xong: STATUS.md; clone NAVSIM main (0a380a9) vào ~/navsim_workspace/navsim; env navsim_v2 (python 3.9, torch 2.0.1, numpy 1.23.4); tải maps + navsim_logs/mini (chưa tải sensor_blobs mini); metric cache mini xong (exp/metric_cache)
- Handoff đã xử lý: (chưa có)
- Vướng mắc / lỗi đã sửa:
  - `pip install nuplan-devkit @ git+...` lỗi "could not create 'build/lib/docs'" do file `BUILD` trùng thư mục `build` trên macOS (filesystem không phân biệt hoa thường). Cách xử lý: clone nuplan-devkit (ce3c323) vào ~/navsim_workspace/dl/nuplan-devkit, xóa file BUILD trong bản clone đó, cài từ đường dẫn cục bộ (requirements: dl/req_local.txt). Không sửa code NAVSIM.
  - run_pdm_score_one_stage cần metric cache trước: chạy run_metric_caching.py train_test_split=mini.

## Việc tiếp theo (người dùng quyết định)
