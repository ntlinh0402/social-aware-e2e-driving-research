# STATUS

## LAPTOP
- Trạng thái máy: (chưa cập nhật) — cập nhật lần cuối: —
- Đang làm:
- Đã xong:
- Handoff mới nhất:
- Vướng mắc:

## MAC
- Trạng thái máy: Darwin arm64, Apple M5, RAM 16 GB, ổ trống ~800 GiB; Miniforge ở ~/miniforge3 (chưa `conda init`, gọi bằng đường dẫn đầy đủ); chưa có tmux — cập nhật lần cuối: 2026-10-28
- Đang làm: A1 — metric caching split mini (chạy nền), sau đó chạy ConstantVelocity non_reactive / reactive
- Đã xong: STATUS.md; clone NAVSIM main (0a380a9) vào ~/navsim_workspace/navsim; env navsim_v2 (python 3.9, torch 2.0.1, numpy 1.23.4); tải maps + navsim_logs/mini (chưa tải sensor_blobs mini)
- Handoff đã xử lý: (chưa có)
- Vướng mắc / lỗi đã sửa:
  - `pip install nuplan-devkit @ git+...` lỗi "could not create 'build/lib/docs'" do file `BUILD` trùng thư mục `build` trên macOS (filesystem không phân biệt hoa thường). Cách xử lý: clone nuplan-devkit (ce3c323) vào ~/navsim_workspace/dl/nuplan-devkit, xóa file BUILD trong bản clone đó, cài từ đường dẫn cục bộ (requirements: dl/req_local.txt). Không sửa code NAVSIM.
  - run_pdm_score_one_stage cần metric cache trước: chạy run_metric_caching.py train_test_split=mini.

## Việc tiếp theo (người dùng quyết định)
