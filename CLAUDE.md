# CLAUDE.md — Social Burden of RL Post-Trained End-to-End Driving Planners

File này hướng dẫn Claude (Claude Code) làm việc trong repo nghiên cứu `research/`. Repo được dùng trên **hai máy**. Đọc toàn bộ file, **xác định mình đang ở máy nào (mục 2)**, rồi chỉ làm phần việc của máy đó.

## 1. Bối cảnh dự án

Nghiên cứu xem **RL post-training có làm planner end-to-end (E2E) "chèn ép" xe khác hơn không** (ép xe khác phanh gấp, rút ngắn khoảng an toàn), đo bằng thước đo phản thực tế so với người lái thật, rồi đề xuất Constrained GRPO có ràng buộc social.

- Benchmark: **NAVSIM** (v1 cho PDMS, v2 cho EPDMS + reactive traffic agents bằng IDM).
- Cặp mô hình so sánh chính (cùng kiến trúc, chỉ khác RL):
  - **DiffusionDrive (IL)**: checkpoint `diffusiondrive_navsim_88p1_PDMS`
  - **DiffusionDriveV2 (RL)**: `diffusiondrivev2_rl.ckpt` (và `diffusiondrivev2_sel.ckpt` = RL + mode selector)
- Mốc tham chiếu: Human (quỹ đạo log), PDM-Closed (rule-based).
- Không có server GPU. Toàn bộ Milestone 0 chạy trên hai máy cá nhân.

## 2. Hai máy và vai trò

Ý tưởng chính: **tách bước sinh quỹ đạo (cần GPU NVIDIA) khỏi bước mô phỏng và chấm điểm (chỉ cần CPU và RAM)**. Hai máy trao đổi với nhau bằng **file quỹ đạo** qua Git.

| | **LAPTOP** | **MAC** |
|---|---|---|
| Phần cứng | ASUS TUF FX705, Ryzen 7 3750H (4C/8T), GPU NVIDIA GTX (4–6 GB VRAM), **RAM 8 GB (không nâng được)**, SSD + HDD 1 TB | MacBook Pro M5 (Apple Silicon, không có CUDA) |
| Hệ điều hành | Ubuntu 22.04 (dual boot) | macOS |
| Vai trò | **Chạy mô hình → xuất quỹ đạo** | **Mô phỏng NAVSIM, IDM, chấm PDMS/EPDMS, tính chỉ số social, phân tích, viết báo cáo** |
| Env conda | `ddv2` | `navsim_v1` (NAVSIM nhánh `v1.1`, chấm PDMS), `navsim_v2` (NAVSIM `main`, reactive IDM) |
| Dữ liệu | maps + split `test` đầy đủ (logs **và** sensor_blobs), đặt trên HDD | maps + `navsim_logs` của split `test` (thử **không** tải sensor_blobs) + split `mini` để thử cài |
| KHÔNG làm | Chấm điểm toàn bộ navtest, metric caching quy mô lớn | Chạy mô hình mạng nơ-ron |

**Xác định máy khi bắt đầu phiên làm việc:**
```bash
uname -s          # Linux → LAPTOP ; Darwin → MAC
nvidia-smi        # chỉ có trên LAPTOP
```
In kết quả ra, ghi vào `STATUS.md`, rồi chỉ làm các bước có nhãn đúng máy (`[LAPTOP]` hoặc `[MAC]`).

## 3. Đồng bộ hai máy

### 3.1 Cái gì đồng bộ, cái gì không

| Đồng bộ qua Git (repo `research/`, private trên GitHub) | KHÔNG đồng bộ (mỗi máy tự có) |
|---|---|
| `CLAUDE.md`, `STATUS.md`, `scripts/`, `configs/` | Dataset (maps, logs, sensor_blobs) |
| `handoff/` (file quỹ đạo, token list, manifest) | Checkpoint mô hình |
| `results/` (CSV, REPORT.md, hình) | Metric cache, feature cache (`exp/`) |
| `logs/` (chỉ log tóm tắt, không log khổng lồ) | Môi trường conda |

File quỹ đạo rất nhỏ (mỗi scene chỉ vài pose), nên commit thẳng vào Git được. Nếu một file vượt 50 MB thì không commit: chuyển bằng `rsync`/`scp` trong mạng LAN hoặc ổ đĩa chung, và ghi đường dẫn cùng checksum vào manifest.

### 3.2 Quy tắc Git

- **Đầu phiên:** `git pull --rebase`. Đọc `STATUS.md` để biết việc tiếp theo của máy mình.
- **Cuối mỗi bước:** cập nhật `STATUS.md`, rồi `git add` → `git commit -m "[LAPTOP] ..."` hoặc `"[MAC] ..."` → `git push`.
- Mỗi máy chỉ ghi vào phần của mình trong `STATUS.md` và các thư mục con theo máy (`handoff/` do LAPTOP ghi, `results/` do MAC ghi), để tránh xung đột.
- Nếu `git pull` báo xung đột, dừng lại và hỏi người dùng. Không force push.

### 3.3 Hợp đồng bàn giao (handoff contract)

Mỗi lần LAPTOP xuất quỹ đạo, tạo một thư mục:

```
handoff/<run_id>/            # run_id = <model>_<split>_<subset>_<YYYYMMDD-HHMM>
├── trajectories.pkl         # dict: token -> {"poses": np.ndarray (N, 3) [x, y, heading], local ego frame}
├── tokens.txt               # danh sách token đã chạy, mỗi dòng một token
└── manifest.json
```

`manifest.json` bắt buộc có các trường:
```json
{
  "run_id": "...",
  "model": "diffusiondrive_il | diffusiondrivev2_rl | diffusiondrivev2_sel",
  "checkpoint_file": "...",
  "checkpoint_sha256": "...",
  "split": "navtest",
  "subset": "all | chunk_03 | pilot_seed42_n1000 | ...",
  "num_tokens": 0,
  "trajectory_sampling": {"num_poses": 8, "interval_length": 0.5},
  "repo_commits": {"DiffusionDriveV2": "...", "research": "..."},
  "device": "cuda (GTX ...)",
  "batch_size": 1,
  "created_at": "..."
}
```
Giá trị `trajectory_sampling` phải lấy đúng từ code mô hình. Hãy kiểm tra, đừng giả định.

**MAC trước khi chạy một handoff phải:** kiểm tra mọi token trong `tokens.txt` có tồn tại trong scene loader của mình; kiểm tra số pose khớp với `trajectory_sampling`; nếu lệch thì dừng và ghi vào `STATUS.md`.

### 3.4 `STATUS.md`, bảng điều phối chung

Tạo ở gốc repo nếu chưa có, theo mẫu:

```markdown
# STATUS

## LAPTOP
- Trạng thái máy: (GPU, RAM, swap, ổ trống) — cập nhật lần cuối: ...
- Đang làm:
- Đã xong:
- Handoff mới nhất: handoff/<run_id>
- Vướng mắc:

## MAC
- Trạng thái máy: ...
- Đang làm:
- Đã xong:
- Handoff đã xử lý: ...
- Vướng mắc:

## Việc tiếp theo (người dùng quyết định)
```

## 4. Cấu trúc thư mục trên mỗi máy

```
~/navsim_workspace/
├── navsim/            # MAC: NAVSIM main (env navsim_v2) ; có thể clone thêm navsim_v1 (nhánh v1.1)
├── navsim_v1/         # MAC: git clone -b v1.1 (env navsim_v1)
├── DiffusionDriveV2/  # LAPTOP: repo mô hình (có bản navsim riêng bên trong) + ckpts/
├── dataset/           # mỗi máy tự có; LAPTOP đặt trên HDD và symlink về đây
├── exp/               # NAVSIM_EXP_ROOT, cache cục bộ
└── research/          # repo Git đồng bộ (CLAUDE.md, STATUS.md, scripts, handoff, results)
```

Biến môi trường (chỉnh `NAVSIM_DEVKIT_ROOT` theo env đang dùng):
```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/<repo của env đang active>"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```
Trước mỗi lệnh quan trọng: in env đang active và `NAVSIM_DEVKIT_ROOT`.

## 5. Quy tắc riêng cho LAPTOP (RAM 8 GB)

- Dùng Ubuntu nhẹ, tắt mọi ứng dụng không cần thiết khi chạy.
- Tạo **swap 16–32 GB trên SSD** (không đặt trên HDD). Có thể bật thêm `zram`. Ghi cấu hình vào `STATUS.md`.
- Inference **batch size = 1**, chạy theo **chunk khoảng 500 token**; mỗi chunk là một handoff riêng, hoặc gộp sau khi xong.
- Dataloader dùng ít worker (0–2). Theo dõi `free -h` và `nvidia-smi`. Nếu bị OOM thì giảm chunk hoặc worker trước khi thử cách khác.
- Kiểm tra sớm: phiên bản PyTorch mà repo yêu cầu có hỗ trợ GPU của máy không (`torch.cuda.is_available()`, compute capability).
- Không chạy metric caching cho toàn bộ navtest trên LAPTOP.

## 6. Quy trình Milestone 0 (theo thứ tự)

Đánh dấu `[x]` khi xong. Lưu stdout/stderr vào `logs/<bước>_<timestamp>.log`.

**Giai đoạn A — Cài đặt song song**
- [ ] **A1 [MAC]** Cài env `navsim_v2` (NAVSIM `main`). Tải `download_maps` và `download_mini`. Chạy `run_pdm_score_one_stage.py` với agent ConstantVelocity trên split mini, lần lượt `traffic_agents=non_reactive` và `traffic_agents=reactive`. Nếu cài không được trên macOS, ghi chi tiết lỗi vào `STATUS.md` và dừng để người dùng quyết định.
- [ ] **A2 [MAC]** Cài env `navsim_v1` (nhánh `v1.1`), chạy thử ConstantVelocity trên mini để có PDMS v1.
- [ ] **A3 [LAPTOP]** Kiểm tra GPU/driver/RAM/ổ cứng, tạo swap. Cài env `ddv2` theo `DiffusionDriveV2/docs/install.md`. Tải checkpoint vào `DiffusionDriveV2/ckpts/`: `resnet34.a1_in1k`, `diffusiondrive_navsim_88p1_PDMS`, `diffusiondrivev2_rl.ckpt`, `diffusiondrivev2_sel.ckpt`.
- [ ] **A4 [LAPTOP]** Tải `download_maps` + `download_test` lên HDD (hỏi người dùng trước khi tải, báo dung lượng ổ trống).
- [ ] **A5 [MAC]** Tải `navsim_logs` của split test (không sensor_blobs). Nếu script tải không cho tách riêng logs, báo người dùng.

**Giai đoạn B — Thử nghiệm nhỏ (50 scene) và kiểm tra chéo**
- [ ] **B1 [LAPTOP]** Viết `scripts/export_trajectories.py`: nạp agent và checkpoint của repo DiffusionDriveV2, chạy trên một danh sách token, xuất handoff đúng định dạng mục 3.3. Không viết lại mô hình, chỉ gọi lại agent và feature builder có sẵn. Chạy cho 50 token của navtest với cả 3 checkpoint, rồi push.
- [ ] **B2 [MAC]** Viết `scripts/replay_agent.py`: lớp `ReplayAgent` kế thừa `navsim.agents.abstract_agent.AbstractAgent`, không dùng cảm biến, `compute_trajectory()` trả về quỹ đạo theo token từ `trajectories.pkl`. Đặt sao cho dùng được ở cả hai env `navsim_v1` và `navsim_v2`.
- [ ] **B3 [MAC]** Chấm PDMS (env `navsim_v1`) cho 50 scene ở B1. **Kiểm tra chéo:** nếu LAPTOP chấm được PDMS cho đúng 50 scene này bằng script của repo DiffusionDriveV2 (metric cache nhỏ, vừa RAM), hai kết quả phải trùng nhau. Nếu lệch, tìm nguyên nhân (frame tọa độ, sampling, token) trước khi đi tiếp.

**Giai đoạn C — Toàn bộ navtest**
- [ ] **C1 [LAPTOP]** Xuất quỹ đạo toàn bộ navtest cho 3 checkpoint, theo chunk, rồi push từng handoff.
- [ ] **C2 [MAC]** Metric caching navtest (env `navsim_v1`) và chấm PDMS cho cả 3 handoff. So với số công bố (mục 7).
- [ ] **C3 [MAC]** Chạy NAVSIM v2 one-stage cho các handoff với `non_reactive` và `reactive`; chạy thêm agent Human và PDM-Closed làm mốc. Ghi chênh lệch điểm.
- [ ] **C4 [MAC]** Viết `results/REPORT.md` (mẫu mục 8), push.

## 7. Con số kỳ vọng (để so khớp)

| Mô hình | Benchmark | Công bố |
|---|---|---|
| DiffusionDrive (ResNet-34) | NAVSIM v1 navtest, PDMS | 88,1 |
| DiffusionDriveV2 (ResNet-34) | NAVSIM v1 navtest, PDMS | 91,2 |
| DiffusionDriveV2 (ResNet-34) | NAVSIM v2, EPDMS | 85,5 |

- Chênh lệch ≤ 0,5 điểm thì coi là khớp. Nếu lớn hơn: kiểm tra checkpoint, split, phiên bản NAVSIM, và frame/sampling của quỹ đạo trước khi kết luận.
- Xác định trong paper xem con số 91,2 ứng với `_rl` hay `_sel`.
- Nếu không chấm được toàn bộ navtest, báo kết quả trên tập con (ghi rõ seed và số scene). Khi đó **không** so trực tiếp với số công bố.

## 8. Mẫu `results/REPORT.md`

```markdown
# Milestone 0 — Tái hiện baseline (2 máy)

## Máy và môi trường
- LAPTOP: GPU / RAM / swap / ổ — env ddv2 (python, torch, CUDA)
- MAC: chip / RAM — env navsim_v1, navsim_v2
- Commit: navsim (main), navsim_v1 (v1.1), DiffusionDriveV2, research

## Kiểm tra chéo 50 scene (B3)
| Mô hình | PDMS LAPTOP (script repo) | PDMS MAC (ReplayAgent) | Chênh lệch |

## Kết quả navtest
| Mô hình | Checkpoint | Handoff | PDMS tự chạy | Công bố | Chênh lệch |

## NAVSIM v2 one-stage
| Planner | EPDMS non_reactive | EPDMS reactive |

## Thời gian chạy từng bước (mỗi máy)
## Lỗi gặp phải và cách sửa (kèm diff nếu sửa code thư viện)
## Câu hỏi cho người dùng / giảng viên
```

## 9. KHÔNG làm trong Milestone 0

- Không train hay fine-tune mô hình.
- Không tải `trainval` / `navtrain`.
- Không sửa code thư viện của NAVSIM hay DiffusionDriveV2, trừ khi tài liệu yêu cầu; nếu buộc phải sửa (ví dụ đổi device), ghi diff vào REPORT.
- Không cài chung các repo vào một env.
- Không tự đoán con số. Mọi số liệu phải trích từ CSV/log, ghi rõ đường dẫn.
- Không tải dataset lớn hoặc xóa dữ liệu khi chưa hỏi người dùng.

## 10. Quy tắc làm việc chung

- Lệnh dài chạy nền (`tmux`), có log, và báo tiến độ ngắn gọn.
- Gặp lỗi: đọc traceback, kiểm tra env và biến môi trường trước, rồi mới sửa. Đổi phiên bản package thì ghi lý do.
- Script phụ trợ đặt trong `scripts/`, có docstring và tham số dòng lệnh rõ ràng.
- Cuối mỗi phiên: cập nhật `STATUS.md`, rồi commit và push.

## 11. Milestone tiếp theo (CHƯA làm — chỉ để hiểu hướng đi)

Dùng lại đúng quy trình hai máy (LAPTOP xuất quỹ đạo, MAC mô phỏng):

1. **[MAC] Hook ghi trạng thái xe IDM**: tìm module traffic agents (config `traffic_agents_policy`) và ghi vận tốc/gia tốc từng xe theo thời gian khi chạy one-stage với `traffic_agents=reactive`.
2. **[MAC] Chỉ số social phản thực tế**, cho scene s, xe j, planner π:
   - `d_j(π)` = mức giảm tốc lớn nhất của xe j trong 4 s khi ego đi theo π
   - `B_j(π) = max(0, d_j(π) − d_j(Human))`, `C_social = Σ_j B_j`
   - IHBR = % scene có ít nhất 1 xe giảm tốc > 3 m/s² trong khi với Human thì không (báo thêm ngưỡng 2 và 4 m/s²)
   - Kiểm tra: với Human, mọi chỉ số phải bằng 0.
3. **[MAC] Pilot H1** trên tập con navtest (seed cố định): so sánh ghép cặp IL và RL (Wilcoxon, McNemar, bootstrap CI), tương quan ΔEP với ΔC_social.
4. Giai đoạn train (Constrained GRPO + social) cần GPU mạnh hơn: thuê theo đợt ngắn, ngoài phạm vi hai máy này.

## 12. Tài liệu

- NAVSIM: https://github.com/autonomousvision/navsim (docs: install.md, agents.md, traffic_agents.md, metrics.md, splits.md; nhánh `v1.1` cho NAVSIM v1)
- DiffusionDriveV2: https://github.com/hustvl/DiffusionDriveV2 (docs/install.md, docs/train_eval.md)
- Checkpoints: https://huggingface.co/hustvl/DiffusionDriveV2, https://huggingface.co/hustvl/DiffusionDrive
- Paper: NAVSIM v2 (arXiv 2506.04218), DiffusionDrive (arXiv 2411.15139), DiffusionDriveV2 (arXiv 2512.07745), Constrained GRPO (arXiv 2602.05863)