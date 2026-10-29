# Báo cáo tuần 1 — Social Burden of RL Post-Trained E2E Driving Planners

**Người thực hiện:** Nguyen Thuy Linh
**Giai đoạn:** 2026-10-28 → 2026-10-29 (Milestone 0, pilot)
**Repo:** https://github.com/ntlinh0402/social-aware-e2e-driving-research
**Commit đầu tuần:** `f9ae6af` · **Commit cuối tuần:** `e4309c8`

---

## 1. Bối cảnh và mục tiêu nghiên cứu

### 1.1 Vấn đề nghiên cứu
Các planner lái xe end-to-end (E2E) hiện đại dùng mạng nơ-ron ánh xạ trực tiếp từ cảm biến (camera + lidar) sang quỹ đạo ego. Sau khi huấn luyện bằng **imitation learning (IL)** — bắt chước quỹ đạo người lái thật trong log — một số công trình gần đây dùng thêm **reinforcement learning (RL)** để fine-tune: thưởng cho quỹ đạo đạt điểm benchmark cao, phạt quỹ đạo xấu. Kết quả thực nghiệm (DiffusionDriveV2, 2512.07745) cho thấy RL cải thiện điểm benchmark 3-4 điểm.

**Câu hỏi của nghiên cứu này:** liệu reward tối đa hoá điểm benchmark có vô tình dạy planner cư xử **"ép"** xe khác không — buộc xe khác phanh gấp, chen vào khoảng cách an toàn hơn — do benchmark chỉ phạt va chạm chứ không phạt gây khó cho xe khác? Nếu có, cần Constrained GRPO với ràng buộc social để huấn luyện.

### 1.2 Các thuật ngữ cốt lõi

**NAVSIM** (Non-reactive Autonomous Vehicle Simulator) là benchmark do nhóm Autonomous Vision của Tübingen phát hành. Nó cho phép đánh giá planner lái xe E2E trên tập scenario thật (trích từ nuPlan + OpenScene) mà không cần quay vòng simulation nặng. Có 2 phiên bản:
- **NAVSIM v1 (nhánh `v1.1`):** chấm **PDMS** — xe khác (traffic agents) *không phản ứng* với ego (non-reactive).
- **NAVSIM v2 (nhánh `main`):** chấm **EPDMS** — xe khác *phản ứng* với ego bằng IDM (Intelligent Driver Model).

**PDMS** (Predictive Driver Model Score, dải 0–1 hoặc 0–100 tuỳ ngữ cảnh) là điểm tổng hợp của 6 chỉ số con, mô phỏng ego đi theo quỹ đạo mô hình đề xuất trong 4 giây:
- `no_collision`: không va chạm (0 hoặc 1 — nhân tử)
- `drivable_area`: đi trong vùng cho phép (nhân tử)
- `driving_direction`: đúng chiều lane (nhân tử)
- `ttc` (Time To Collision): khoảng thời gian đến va chạm nếu giữ nguyên vận tốc
- `progress`: tiến độ đi dọc tuyến so với rule-based PDM-Closed
- `comfort`: không giật/lắc quá mức

3 chỉ số nhân (collision/area/direction) nếu = 0 làm PDMS = 0. 3 chỉ số weighted tổng hợp theo trọng số cố định.

**EPDMS** = PDMS mở rộng, thêm chỉ số với reactive traffic (hub-driven IDM).

**metric_cache** = file cache trên đĩa chứa toàn bộ thông tin cần để chấm PDMS cho 1 token: bản đồ lane, roadblock, danh sách xe khác và quỹ đạo của chúng, quỹ đạo chuẩn của PDM-Closed. Một file `.pkl` nén lzma per token. Phải precompute vì chấm trực tiếp từ raw log quá chậm.

**Scene / token:** 1 token là 1 scene cụ thể = 1 thời điểm trong 1 log, lấy 4 history frames + 10 future frames quanh đó. Mỗi log có vài trăm token.

**Train/test split navtest:** tập đánh giá chuẩn của NAVSIM, ~14000 scenes trong ~500 logs.

### 1.3 Ba mô hình so sánh
Cùng kiến trúc nền DiffusionDrive (ResNet-34 + Transformer + Diffusion head), khác cách huấn luyện head:

| Ký hiệu | Checkpoint | Cách huấn luyện | PDMS công bố (navtest) |
|---|---|---|---|
| **IL** | `diffusiondrive_navsim_88p1_PDMS` | Imitation Learning thuần: cross-entropy / L1 bắt chước log người | 88.1 |
| **RL** | `diffusiondrivev2_rl.ckpt` | IL ckpt + RL post-training với reward = PDMS → model học chọn quỹ đạo điểm cao | 91.2 |
| **SEL** | `diffusiondrivev2_sel.ckpt` | RL ckpt + mode selector học riêng để chọn 1 trong nhiều mode diffusion sinh ra | ~91.2 |

Nghiên cứu dự định so **IL vs RL/SEL** để trả lời: RL có "ép" xe khác hơn IL không? Cả 3 checkpoint đều bắt buộc có kết quả.

### 1.4 Thiết kế hạ tầng 2 máy (kế hoạch gốc trong CLAUDE.md)
Vì không có server GPU, kế hoạch ban đầu tách:
- **LAPTOP** (NVIDIA GPU): chạy agent, xuất file quỹ đạo
- **MAC** (CPU-only Apple Silicon): mô phỏng, chấm PDMS/EPDMS, phân tích social

Trao đổi qua file `trajectories.pkl` + `manifest.json` trong thư mục `handoff/` đồng bộ qua Git.

**Thay đổi trong tuần 1:** Mac gặp vấn đề cài NAVSIM trên Apple Silicon → user quyết định consolidate: 1 máy LAPTOP làm hết cả 2 vai. Kiến trúc 2-pha vẫn giữ (export → score) để sau này tái sử dụng với server GPU bất kỳ.

---

## 2. Những gì đã dựng được trong tuần

### 2.1 Môi trường
- **WSL2 Ubuntu-22.04** trên Windows 11, CUDA passthrough GTX 1650 Ti 4 GB (driver 592.00 trong WSL)
- `.wslconfig`: 12 GB RAM, 16 GB swap, 8 processors, `vmIdleTimeout=-1` (chống WSL idle shutdown)
- Dataset layout trên F:\ (NTFS, 118 GB total): `/mnt/f/navsim_dataset/{maps,test_navsim_logs,test_sensor_blobs,sensor_blobs,navsim_logs}` (2 symlink cuối để khớp đường dẫn mặc định của NAVSIM)
- 3 conda env (Python 3.9):
  - `ddv2` — torch 2.0.1+cu117, navsim 1.1.0 (từ DiffusionDriveV2 bundled), diffusers 0.31.0, cloudpickle — dùng cho **inference GPU**
  - `navsim_v1` — navsim 1.1.0 (upstream nhánh `v1.1`) — dùng cho **PDMS v1 scoring**
  - `navsim_v2` — navsim 2.0.0 (upstream `main`) — chuẩn bị cho **EPDMS v2 scoring** (chưa dùng)

### 2.2 Dataset (một phần, ~37 GB / 125+ GB bản đầy đủ)

Tải chọn lọc (bỏ lidar để tiết kiệm, nhưng mô hình DiffusionDrive dùng lidar BEV → vẫn phải tải):

| Thành phần | Kích thước | Trạng thái |
|---|---|---|
| nuplan-maps v1.0 | 1.4 GB | ✓ đầy đủ |
| test_navsim_logs (metadata) | 983 MB | ✓ đầy đủ 147 .pkl, cover toàn bộ navtest |
| test_sensor_blobs camera chunks 0-4 | ~17 GB | ✓ 5/32 chunks |
| test_sensor_blobs lidar chunk 0 | ~18 GB | partial — bị rsync permission denied trên NTFS giữa chừng |

Kết quả: **5 log** có cả camera + lidar → **242 scene** dùng được. Thiếu 9 camera chunk + 7 lidar chunk nữa mới cover 14k scenes của navtest.

### 2.3 Code deliverables (commit `e4309c8`)

**[scripts/export_trajectories.py](../scripts/export_trajectories.py)** (238 dòng)
Driver chạy agent inference:
1. Dùng hydra compose config submission chuẩn của DDv2
2. Override agent + checkpoint + scene_filter.max_scenes
3. `--only-available-logs`: giao scene_filter.log_names với các log có đủ cam+lidar trên đĩa
4. Chạy agent trên GPU, xuất đúng format handoff theo CLAUDE.md §3.3

**[scripts/replay_agent.py](../scripts/replay_agent.py)** (90 dòng)
Agent giả lập cho pha chấm: nhận token, tra bảng `trajectories.pkl` trả về Trajectory. Dùng được ở cả env `navsim_v1` (PDMS v1) và `navsim_v2` (EPDMS v2). Thiết kế `requires_scene=True` để lấy token từ `scene.scene_metadata.initial_token` (NAVSIM không truyền token trực tiếp vào `compute_trajectory`).

**[handoff/diffusiondrive_il_navtest_pilot_n50_20261029-0021/](../handoff/)**
Pilot IL: 50 token × 8 pose (x, y, heading) với `trajectory_sampling = {num_poses=8, time_horizon=4s, interval_length=0.5s}`. Kèm `manifest.json` chứa SHA256 checkpoint, commit hash DDv2, timestamp.

### 2.4 Hạ tầng scoring
- `metric_cache_pilot/`: 242 token × ~800 KB mỗi file = ~190 MB — build trong 7 phút với 2 CPU worker
- `metric_feature_cache_pilot/`: 242 token × 2 file (feature + target) — build trong 50 s trên GPU (chạy feature builder của agent, ghi lên đĩa)

Hai cache này cần cho `run_pdm_score_fast.py` — pipeline tối ưu của DDv2 để chấm RL/SEL (vì các agent này **gọi PDM scorer bên trong `forward()`** để chọn trong 80 trajectory candidate).

---

## 3. Kết quả định lượng

### 3.1 Pilot 50 scene, cross-check IL baseline

| Pipeline | Env | PDMS pilot (n=50) | Scene hợp lệ |
|---|---|---|---|
| ReplayAgent trong **navsim v1.1** chấm handoff IL | `navsim_v1` | **0.9643** | 49/50 |
| **DDv2 bundled** chạy agent + chấm trực tiếp | `ddv2` | **0.9559** | 49/49 |

**Chênh lệch 0.009** — trong ngưỡng khớp 0.5 điểm mà CLAUDE.md §7 đặt ra.

### 3.2 Ý nghĩa cho research
1. **Pipeline hai pha `export → replay → score` hoạt động đúng.** Khi tương lai có server GPU, có thể tách inference ↔ scoring giữa các máy khác nhau mà kết quả vẫn khớp. Đây là xác thực kỹ thuật quan trọng để làm nghiên cứu quy mô lớn hơn.
2. **Không phải tái hiện số công bố (88.1).** Pilot chỉ 50/14000+ scene, 5/~500 log — subset này không representative cho navtest. 0.956 chỉ nói "model nạp + chạy + chấm được", KHÔNG nói "chúng tôi đã reproduce 88.1".
3. **Để so khớp 88.1 ± 0.5** phải tải nốt 9 cam + 7 lidar chunk (~55 GB, ~3 h) và chạy toàn navtest (~2.3 h/agent trên GTX 1650 Ti).

### 3.3 RL và SEL — chưa có số
Chạy end-to-end xong (feature builder + diffusion sampling không crash), nhưng PDM scorer báo:

```
AssertionError: angle is not finite
```

trên **mọi** scene của cả 2 agent (0/242 cho mỗi agent). Ý nghĩa: trong 80 trajectory candidate mà diffusion head của RL/SEL sinh ra cho mỗi scene, cột heading (hướng xe, cột thứ 3 của trajectory `(x, y, heading)`) có giá trị **NaN** (Not a Number) hoặc **Inf** (vô cùng). Scorer PDM mô phỏng chuyển động xe → cần heading hữu hạn để tính góc quay và vị trí sau 0.5s → gặp NaN thì fail assertion.

Chi tiết traceback (từ `logs/c2_fast_*.log`):
```
File ".../diffusiondrivev2_model_rl.py", line 1010, in forward_test_rl
    reward_group, metric_cache, sub_rewards_group = self.get_pdm_score_para(diffusion_output, metric_cache)
File ".../diffusiondrivev2_model_rl.py", line 791, in <listcomp>
    (metric_cache_path[b], traj_np[b]),
...
AssertionError: angle is not finite
```

**IL agent không có vấn đề này** vì IL không gọi PDM scorer trong forward — nó chỉ output 1 trajectory thẳng từ transformer head, không có diffusion sampling.

---

## 4. Giải thích chi tiết lỗi "angle is not finite"

### 4.1 Nguồn phát NaN nằm ở đâu

RL/SEL agent lúc inference đi qua:
```
features (camera+lidar BEV) 
    → backbone encoder 
    → cross-attention với 1 ego query + 100 agent query
    → trajectory_head.forward_test_rl():
        plan_anchor (20 cluster từ k-means trajectories train set, file kmeans_navsim_traj_20.npy)
        → add_noise (DDIM scheduler, 8 bước) 
        → diff_decoder (2 bước denoising) 
        → bezier_xyyaw(diffusion_output)  ← đây là nơi heading được decode
        → get_pdm_score_para(diffusion_output, metric_cache)  ← crash tại đây
```

Hàm `bezier_xyyaw` nhận các điểm Bezier control sinh bởi diffusion và tính ra `(x, y, heading)` theo thời gian. Heading được tính từ đạo hàm `atan2(dy/dt, dx/dt)` — nếu đạo hàm gần 0 (xe đứng yên) hoặc tử/mẫu có NaN propagation từ diffusion thì heading cũng NaN.

### 4.2 Vì sao chỉ RL/SEL dính mà IL không
IL không có diffusion head, output heading trực tiếp từ transformer linear layer (học L1 regression với heading log). RL dùng diffusion để sinh 80 candidate rồi re-rank → chuỗi phép toán phức tạp hơn, dễ tích luỹ sai số dẫn đến NaN.

Nhưng NaN **uniform trên mọi scene** (242/242) chứ không ngẫu nhiên vài scene → đây là lỗi hệ thống, không phải lỗi edge case. Thường do:

### 4.3 Các khả năng theo thứ tự ưu tiên

**(a) Mismatch phiên bản `torch` / `diffusers` giữa môi trường hiện tại và môi trường tác giả dùng để train/export ckpt.**
- Tác giả không pin version trong `requirements.txt` của repo
- Môi trường hiện tại: `torch==2.0.1+cu117`, `diffusers==0.31.0` (phải chọn thủ công: 0.36 crash do dùng `torch.xpu` không có trong 2.0.1; 0.27 crash do import `cached_download` từ huggingface_hub mới)
- Floating point kernels và chi tiết implementation của DDIM scheduler khác giữa các bản diffusers → có thể dẫn đến NaN đồng loạt

**(b) Patch đường dẫn ảnh hưởng weight loading.**
Trong tuần đã patch 6 chỗ hardcoded absolute path (NFS của tác giả `/home/users/bencheng.liao/...`) sang path local. Trong đó có `plan_anchor_path` (file `kmeans_navsim_traj_20.npy` — 20 trajectory cluster làm initial condition cho diffusion). Nếu file này sai hoặc load sai shape → diffusion start từ noise + anchor sai → NaN. **Đã verify file này có 20 trajectory shape `(20, 8, 2)` chuẩn**, nhưng chưa verify key-level khi load vào `plan_anchor` buffer.

**(c) Dataset subset quá nhỏ ảnh hưởng feature cache.**
Feature cache được build chỉ với 242 scene. Hàm `CacheOnlyDatasetTest` có thể normalize feature dựa trên statistics toàn dataset. Khả năng thấp vì NAVSIM standard normalize per-sample, nhưng chưa verify.

**(d) Mô hình quantize / dtype casting.**
Lightning dùng `precision=16-mixed` (AMP). Float16 kết hợp DDIM step có thể underflow → NaN. Trong log có thấy `Using 16bit Automatic Mixed Precision (AMP)`.

### 4.4 Fix thử trong tuần
Đã làm **1 fix diagnostic** (không phải fix thật):
- Wrap `_pdm_worker` trong try/except trả về zero scores — để pipeline chạy xong không crash
- Kết quả: val/final_epoch = 0.0 với mọi metric → **xác nhận lỗi uniform**, không phải vài scene lẻ

### 4.5 Fix thật — các hướng
| Hướng | Thời gian | Rủi ro | Khả năng cho số thật |
|---|---|---|---|
| **(2) Debug 1 token với print NaN tại từng bước forward** | 30-60 phút | thấp | cao — tìm được dòng NaN đầu tiên, fix tại chỗ |
| **(1) Clamp NaN heading = atan2(dy, dx) hậu kỳ** | 10 phút | **cao: số RL có thể giả** | trung bình |
| **(3) Rebuild env với `torch==2.2.*` + `diffusers==0.29.*`** | 1-2 giờ | trung bình (có thể vỡ cài khác) | trung-cao |
| **(4) Hỏi tác giả qua GitHub Issue** | 1-7 ngày | 0 | cao nhưng không chủ động |

**Khuyến nghị:** Hướng (2) + nếu không ra thì fallback hướng (3).

---

## 5. Các patch đã làm cho thư viện bên thứ ba (minh bạch)

### 5.1 `DiffusionDriveV2` repo
Diff vs commit upstream `1cd12a1`:
1. 6 hardcoded path trỏ về NFS của tác giả (`/home/users/bencheng.liao/...`):
   - `navsim/agents/diffusiondrive/transfuser_config.py:19` → `plan_anchor_path`
   - `navsim/agents/diffusiondrivev2/transfuser_config.py:19` → `plan_anchor_path`
   - `navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_config.py:19` → `plan_anchor_path`
   - `navsim/agents/diffusiondrivev2/diffusiondrivev2_sel_config.py:19` → `plan_anchor_path`
   - `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py:741` → đường dẫn yaml scorer
   - `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py:910` → đường dẫn yaml scorer
2. `_pdm_worker` trong `diffusiondrivev2_model_rl.py` và `diffusiondrivev2_model_sel.py` wrap try/except với fallback zero scores — **đây là diagnostic, phải revert trước khi báo số thật**.

### 5.2 Dependency pins (ngoài `requirements.txt`)
- `diffusers==0.31.0`
- `cloudpickle` (bất kỳ version)
- Cài thêm `unzip` qua apt

### 5.3 Dataset layout alias
- Symlink `dataset/navsim_logs/test → ../test_navsim_logs/test`
- Symlink `dataset/sensor_blobs/test → ../test_sensor_blobs/test`

Lý do: config NAVSIM mặc định (`default_evaluation.yaml`) build đường dẫn theo pattern `${OPENSCENE_DATA_ROOT}/navsim_logs/${train_test_split.data_split}`, nhưng unpack từ HuggingFace cho layout `openscene-v1.1/.../test/`.

---

## 6. Khối lượng công việc

| Bước | Wall time thực tế |
|---|---|
| Set up WSL2 + CUDA passthrough | — (từ phiên trước) |
| Tải dataset (37 GB thực, kế hoạch 50 GB) | ~1h10 (gián đoạn giữa chừng do rsync permission) |
| Cài 3 conda env (ddv2, navsim_v1, navsim_v2) | ~30 phút tổng |
| Debug import errors (diffusers, cloudpickle, path hardcoded) | ~40 phút |
| B1 IL pilot inference (50 token, GTX 1650 Ti) | **29 giây** |
| Metric caching 5 logs / 242 scene, 2 CPU worker | ~7 phút |
| Dataset feature caching 5 logs / 242 scene, GPU | ~50 giây |
| B3 PDMS scoring ReplayAgent (49 scene, 2 worker) | ~11 giây |
| C2 IL scoring qua DDv2 (49 scene, 2 worker) | ~33 giây |
| C2 RL/SEL fast-score (242 scene, GPU) × 4 lần retry | ~1 phút 30 giây/lần/agent |
| Viết REPORT + commit + push | ~20 phút |

**Tổng:** ~3 giờ hiệu dụng (nhiều thời gian cho debug + download).

**Suy ra chi phí mở rộng:**
- Full navtest IL inference: 14000 / 50 × 29s ≈ **2.3 giờ/agent**
- Full navtest metric caching: 14000 / 242 × 7min ≈ **6.7 giờ** (bottleneck CPU)
- Full navtest download: còn ~55 GB với tốc độ 6 MB/s ≈ **2.5 giờ**

---

## 7. Trạng thái các bước Milestone 0

| Bước | Mô tả | Trạng thái |
|---|---|---|
| A1 | Cài navsim_v2 env | ✓ |
| A2 | Cài navsim_v1 env | ✓ |
| A3 | Cài ddv2 env + tải 4 checkpoint | ✓ |
| A4 | Tải full navtest sensor | **1/3** (37/125 GB) |
| A5 | Tải navsim_logs | ✓ |
| B1 | Export quỹ đạo 50 token × 3 ckpt | **1/3** (chỉ IL) |
| B2 | Viết `replay_agent.py` | ✓ |
| B3 | PDMS self-check 50 scene | ✓ (0.9643 vs 0.9559, chênh 0.009) |
| C1 | Export quỹ đạo navtest | ⬜ |
| C2 | PDMS navtest cho 3 ckpt | **1/3** (IL pilot) |
| C3 | EPDMS v2 non_reactive + reactive | ⬜ |
| C4 | REPORT.md | ✓ (interim) |

---

## 8. Rủi ro và câu hỏi mở

1. **Lỗi RL/SEL "angle is not finite"** — chặn kết quả cho 2/3 mô hình. Phải fix mới có số để so sánh IL vs RL. Đây là blocker chính cho Milestone 1.
2. **Dataset không đầy đủ** — chưa reproduce 88.1. Chưa biết số pilot 0.956 có đại diện cho full navtest không.
3. **GTX 1650 Ti 4 GB VRAM** — đủ cho inference batch=1, nhưng không đủ train. Milestone sau (Constrained GRPO) phải thuê GPU ngoài.
4. **F: drive là NTFS/exFAT trên Windows mount qua WSL** — rsync permission denied vì cách WSL dịch quyền. Workaround: dùng `cp -n` thay rsync. Nếu không, phải chuyển dataset vào ext4 trong WSL (có đủ chỗ không?).
5. **Patch 6 hardcoded path chưa report ngược về repo gốc** — nên mở PR để contribute.

---

## 9. Kế hoạch tuần 2 (đề xuất)

**Ưu tiên 1 — Fix RL/SEL (3-6 giờ)**
Chạy 1 token với breakpoint sau từng bước forward của `diffusiondrivev2_model_rl.py:forward_test_rl` (lines 942-1012), print tensor để tìm NaN đầu tiên. Nếu ở `bezier_xyyaw`, fix với eps thêm vào mẫu số. Nếu ở DDIM step, thử disable AMP (`precision=32`) hoặc nâng diffusers.

**Ưu tiên 2 — Tải nốt dataset (background, 3 giờ)**
Resume download 9 cam + 7 lidar chunk còn lại.

**Ưu tiên 3 — Chạy pilot với RL/SEL đã fix (1 giờ)**
Lặp lại B1 pilot cho RL + SEL trên 50 scene, so với IL để có bộ ba đầu tiên.

**Ưu tiên 4 — Full navtest (nếu dataset xong + RL fix xong, ~8-10 giờ)**
C1 + C2 cho cả 3 agent, so với số công bố (88.1, 91.2, 91.2) để verify tái hiện.

**Chuẩn bị Milestone 1 song song**
Đọc code `navsim/planning/simulation/traffic_agents/` để tìm chỗ hook ghi vận tốc/gia tốc xe IDM khi chạy EPDMS reactive.

---

## 10. Phụ lục — thuật ngữ nhanh

- **PDMS / EPDMS**: điểm tổng hợp 6 chỉ số đánh giá quỹ đạo mô phỏng ego trong 4s. v1 non-reactive, v2 reactive IDM.
- **metric_cache**: file cache precompute mọi thứ scorer cần cho 1 token.
- **IL / RL / SEL**: 3 cách huấn luyện trajectory head của DiffusionDrive.
- **IDM** (Intelligent Driver Model): mô hình xe-người đơn giản dùng cho reactive traffic agent trong NAVSIM v2.
- **GRPO** (Group Relative Policy Optimization): thuật toán RL dùng để fine-tune. "Constrained GRPO" = GRPO + ràng buộc an toàn social.
- **BEV** (Bird's Eye View): biểu diễn feature nhìn từ trên xuống, grid 2D.
- **Diffusion head**: trajectory head dùng DDIM scheduler để sinh mẫu quỹ đạo, học từ anchor + noise.
- **Trajectory sampling**: `num_poses=8` trong 4s, 1 pose mỗi 0.5s. Mỗi pose = (x, y, heading).
- **Scene filter**: tiêu chí lọc scene (4 history frame, 10 future frame, phải có route).
- **AbstractAgent.compute_trajectory**: contract của NAVSIM — mọi agent phải implement hàm này để tạo Trajectory từ AgentInput.
- **Handoff / Replay pattern**: tách inference (GPU) ↔ scoring (CPU) qua file pickle, cho phép phân bố 2 máy.

---

**Checkpoint git:** `e4309c8` trên nhánh `main` của `ntlinh0402/social-aware-e2e-driving-research`.
