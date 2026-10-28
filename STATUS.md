# STATUS

## LAPTOP
- Trạng thái máy: ASUS TUF A17 FA706II, Windows 11 Home (build 26200), **1 ổ Kingston NVMe SSD 512 GB** (C:42 GB / O:108 GB / F:115 GB trống — **không có HDD** như CLAUDE.md giả định), RAM 16 GB (tốt hơn 8 GB brief đoán), GPU **NVIDIA GTX 1650 Ti 4 GB** (driver 32.0.15.9200) + iGPU AMD Radeon. WSL2 đã bật, chưa cài distro. Repo clone sẵn ở `O:\research`. — cập nhật lần cuối: 2026-10-28
- Đang làm: A3 gần xong (env ddv2 ready, CUDA verified). Còn: tải 4 checkpoint + tải dataset test (A4 — chặn do dung lượng).
- Đã xong:
  - WSL2 Ubuntu-22.04 (kernel 5.15), user `linh` + sudo NOPASSWD, `.wslconfig` giới hạn 12GB RAM + 16GB swap + 8 cores.
  - CUDA passthrough: driver 592.00 trong WSL, `torch.cuda.is_available()=True`, device **GTX 1650 Ti (sm_75)**, torch 2.0.1+cu117.
  - Miniforge3 ở `~/miniforge3`; env **`ddv2`** (python 3.9, pip 23.3.1) với toàn bộ `DiffusionDriveV2/requirements.txt` + `diffusers 0.36.0` + `einops 0.8.2`. nuplan-devkit 1.2.0 import OK.
  - Clone `DiffusionDriveV2` (depth 1) vào `~/navsim_workspace/DiffusionDriveV2`.
  - Layout: `research → /mnt/o/research`, `dataset → /mnt/f/navsim_dataset` (symlink), `exp` + `DiffusionDriveV2` trong WSL ext4.
  - Script `~/navsim_workspace/activate_ddv2.sh` set env vars (NUPLAN_MAP_VERSION, NUPLAN_MAPS_ROOT, NAVSIM_EXP_ROOT, NAVSIM_DEVKIT_ROOT, OPENSCENE_DATA_ROOT) đúng mục 4 CLAUDE.md. Test source: tất cả vars + CUDA OK.
- Handoff mới nhất: (chưa có)
- Vướng mắc:
  - **Dung lượng F: chỉ 115 GB — có thể KHÔNG đủ cho test đầy đủ** (32 chunks camera + 32 chunks lidar ≈ 125–330 GB theo quy mô OpenScene v1.1). Chờ user quyết strategy (chỉ camera / tải một phần / ổ ngoài).
  - Env `ddv2` hiện gộp `navsim` của DiffusionDriveV2 (install.md yêu cầu `conda activate navsim`). NAVSIM_DEVKIT_ROOT trỏ về `DiffusionDriveV2/` (có folder `navsim/` bên trong). Nếu B1 cần navsim upstream riêng, sẽ clone sau.

## MAC
- Trạng thái máy: Darwin arm64, Apple M5, RAM 16 GB, ổ trống ~800 GiB; Miniforge ở ~/miniforge3 (chưa `conda init`, gọi bằng đường dẫn đầy đủ); chưa có tmux — cập nhật lần cuối: 2026-10-28
- Đang làm: A1 — ConstantVelocity mini: non_reactive đã chạy xong 3615/3615 (log logs/A1_cv_non_reactive_20261028-1249.log, chưa đọc điểm); reactive bị gián đoạn khi máy gập (~111/3615), cần chạy lại
- Đã xong: STATUS.md; clone NAVSIM main (0a380a9) vào ~/navsim_workspace/navsim; env navsim_v2 (python 3.9, torch 2.0.1, numpy 1.23.4); tải maps + navsim_logs/mini (chưa tải sensor_blobs mini); metric cache mini xong (exp/metric_cache)
- Handoff đã xử lý: (chưa có)
- Vướng mắc / lỗi đã sửa:
  - `pip install nuplan-devkit @ git+...` lỗi "could not create 'build/lib/docs'" do file `BUILD` trùng thư mục `build` trên macOS (filesystem không phân biệt hoa thường). Cách xử lý: clone nuplan-devkit (ce3c323) vào ~/navsim_workspace/dl/nuplan-devkit, xóa file BUILD trong bản clone đó, cài từ đường dẫn cục bộ (requirements: dl/req_local.txt). Không sửa code NAVSIM.
  - run_pdm_score_one_stage cần metric cache trước: chạy run_metric_caching.py train_test_split=mini.

## Việc tiếp theo (người dùng quyết định)
