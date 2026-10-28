# LAPTOP_BRIEF — việc cần làm đầu tiên trên LAPTOP (viết bởi Claude trên MAC, 2026-10-28)

Đọc `CLAUDE.md` trước. File này chỉ bổ sung những gì MAC đã học được, để LAPTOP làm đúng ngay từ đầu.

## 0. Trước khi làm gì
1. `uname -s; nvidia-smi; free -h; df -h` → ghi vào mục LAPTOP trong `STATUS.md`.
2. `git pull --rebase`. **Chỉ sửa mục LAPTOP trong STATUS.md, thư mục `handoff/`, và `scripts/export_*`.** Không sửa `results/`, mục MAC.
3. Git: máy này phải tự đăng nhập GitHub (SSH key hoặc `gh auth login`). Không ghi token vào file nào trong repo. Không force push. Xung đột → dừng, hỏi người dùng.
4. Chưa có `scripts/` nào cho LAPTOP; `logs/` chỉ commit log tóm tắt (không commit log hàng MB). Thêm `logs/*.log` lớn vào `.gitignore` nếu cần.

## 1. Quy tắc chạy lệnh dài (bài học đắt giá từ MAC)
- Chạy nền (`tmux`/`nohup`), log vào `logs/<bước>_<timestamp>.log`.
- **Không bao giờ báo "xong" chỉ vì thấy dòng log cuối.** Sau mỗi lần chạy phải kiểm tra (a) mã thoát, (b) file kết quả có thật, (c) số lượng khớp kỳ vọng. MAC từng mất hơn 3 giờ vì một script chỉ ghi CSV ở bước cuối rồi crash.
- **Luôn thử mẫu rất nhỏ (20–50 token) chạy đủ từ đầu đến cuối và đọc kết quả**, rồi mới chạy lớn.
- Báo cho người dùng khi máy nóng/hao pin/OOM; không tự chạy tiếp số lượng lớn khi chưa được duyệt.

## 2. Cấu trúc handoff (đúng mục 3.3 CLAUDE.md)
```
handoff/<model>_<split>_<subset>_<YYYYMMDD-HHMM>/
  trajectories.pkl   # dict token -> {"poses": np.ndarray (N,3) [x,y,heading], local ego frame}
  tokens.txt
  manifest.json
```
- `model` ∈ {diffusiondrive_il, diffusiondrivev2_rl, diffusiondrivev2_sel}. `split` = navtest.
- `trajectory_sampling` (num_poses, interval_length) phải **đọc từ code/config của mô hình**, không đoán (NAVSIM chuẩn: 8 pose, 0.5 s).
- `checkpoint_sha256` tính thật (`shasum -a 256`). `repo_commits` ghi commit DiffusionDriveV2 và research.
- Quỹ đạo phải ở **frame ego cục bộ**, đúng kiểu `Trajectory.poses` của NAVSIM. Không đổi sang frame toàn cục.
- `tokens.txt` phải đúng tập token navtest chuẩn (lấy từ scene filter `navtest` của repo mô hình). Pilot 50 scene: chọn seed cố định, ghi seed vào `subset` (vd `pilot_seed42_n50`).
- Dùng chung một tập token pilot cho cả 3 checkpoint để so ghép cặp được.
- File > 50 MB không commit; ghi đường dẫn + sha256 vào manifest.

## 3. Thứ tự
A3 (GPU/driver/RAM/swap, env `ddv2`, checkpoint) → A4 (hỏi người dùng trước khi tải test) → B1 (50 token × 3 checkpoint, push) → báo MAC qua STATUS.md. Chưa chạy toàn bộ navtest (C1) khi người dùng chưa duyệt kết quả B3.

## 4. Ước lượng thời gian (CHƯA đo, chỉ để lập kế hoạch)
- Cài env + tải checkpoint: 1–3 giờ. Tải `test` (logs + sensor_blobs) lên HDD: có thể nhiều giờ đến cả ngày, tùy mạng và dung lượng; **báo dung lượng trước**.
- Inference batch 1 trên GTX 4–6 GB: giả định 0.3–1 s/scene → navtest (~12 nghìn scene) ≈ 1–3.5 giờ mỗi checkpoint; ba checkpoint ≈ 3–10 giờ. Đo thật bằng pilot 50 token rồi cập nhật con số vào STATUS.md.

## 5. Điều MAC đã làm (để biết bối cảnh)
- `navsim_v2` đã cài trên MAC; mini chạy được. Đang chạy ConstantVelocity trên `navmini` với `frame_interval=1` (script `scripts/run_a1_navmini.sh`).
- NAVSIM v2 cần các scene liền kề (`frame_interval=1`, đúng với split `navtest`) mới tổng hợp được điểm; split mini mặc định không có. Với navtest không phải lo việc này.
