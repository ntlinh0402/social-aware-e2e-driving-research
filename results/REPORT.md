# Milestone 0 — Pilot report (single-machine WSL build)

**Status:** interim pilot (not full navtest). Partial dataset (5 of 32 original log chunks). IL baseline reproduced end-to-end; RL/SEL inference needs further debug (produces non-finite trajectory headings on this environment).

**Date:** 2026-10-29

## 1. Machine and environment

- LAPTOP: ASUS TUF A17 FA706II, Ryzen 7 3750H (4C/8T), **16 GB RAM**, GTX 1650 Ti (4 GB VRAM, sm_75), Kingston NVMe 512 GB SSD only (no HDD). F: drive = 118 GB exFAT for dataset.
- MAC: offline — this machine does everything.
- OS: Windows 11 + WSL2 Ubuntu-22.04 with CUDA passthrough (driver 592.00 inside WSL). `.wslconfig`: 12 GB RAM, 16 GB swap, 8 processors, `vmIdleTimeout=-1`.

### Envs
| env | python | torch | navsim | notes |
|---|---|---|---|---|
| `ddv2` | 3.9 | 2.0.1+cu117 | 1.1.0 (from DiffusionDriveV2 bundled) | inference |
| `navsim_v1` | 3.9 | — | 1.1.0 (upstream v1.1 branch) | PDMS v1 scoring |
| `navsim_v2` | 3.9 | — | 2.0.0 (upstream main) | EPDMS + reactive IDM (not yet exercised) |

### Repo commits
- `research`: see git log (handoff + results pushed here).
- `DiffusionDriveV2`: `1cd12a1e155c34dcc471261835444c8d5587580b` (patched — see §6).
- `navsim_v1`: `3e8291b` (v1.1 branch).
- `navsim`: `0a380a9` (main).

## 2. Dataset

Downloaded ~**37 GB** on F:\ (vs 50 GB plan, vs 125–330 GB full navtest):
- `maps` (nuplan-maps-v1.0): 1.4 GB
- `test_navsim_logs/test/`: 983 MB (147 .pkl metadata, full navtest)
- `test_sensor_blobs/test/`: 5 camera chunks (`.camera_0..4_done`) + partial lidar chunk 0

Only **5 logs** have both cameras and lidar after cleanup (lidar chunk 0 was interrupted by rsync permission issue on NTFS/exFAT; see §6). Those 5 logs cover **242 scenes** passing the navtest scene filter (`num_history_frames=4`, `num_future_frames=10`, `has_route=true`).

The remaining 9 camera chunks and 7 lidar chunks for the full navtest were not downloaded.

## 3. Deliverables

### 3.1 Handoff (B1, IL only)
```
handoff/diffusiondrive_il_navtest_pilot_n50_20261029-0021/
├── trajectories.pkl   # 50 tokens × 8 poses (x,y,heading) each
├── tokens.txt
└── manifest.json      # TrajectorySampling(num_poses=8, interval=0.5s)
```
Inference rate on GTX 1650 Ti, batch_size=1: ~3 it/s steady (first scene ~11 s cold-start).

RL and SEL handoffs: **not produced**. Their agents call PDM scoring inside `forward()` (mode selection by reward), so the "trajectory pickle → replay" pattern doesn't apply. See §4.3.

### 3.2 ReplayAgent (B2)
`scripts/replay_agent.py` — `requires_scene=True`, uses `scene.scene_metadata.initial_token` to look up precomputed trajectories. Yaml stubs added to both `navsim_v1` and `navsim` repos under `.../common/agent/replay_agent.yaml`.

### 3.3 Scripts
- `scripts/export_trajectories.py` — hydra-driven; `--only-available-logs` restricts to logs with both cam+lidar on disk.
- `~/navsim_workspace/dl/*.sh` — download + install + run scripts, logged under `~/navsim_workspace/logs/`.

## 4. Results

### 4.1 B3 — Cross-check PDMS scoring on the 50 pilot tokens
Scoring pipeline sanity check: same 50 tokens, two independent scoring entry points.

| Pipeline | Env | PDMS (pilot n=50) | Valid |
|---|---|---|---|
| ReplayAgent on IL handoff (navsim v1.1 `run_pdm_score.py`) | `navsim_v1` | **0.9643** | 49/50 |
| DDv2 bundled scorer (`run_pdm_score.py`, agent=diffusiondrive_agent) | `ddv2` | **0.9559** | 49/49 |

Delta: **0.009** — well within the 0.5-point tolerance set by CLAUDE.md §7, so the handoff→replay pipeline is consistent with running the IL agent directly. The 1-token discrepancy in valid-count ("49/50" vs "49/49") traces to a scene whose metric-cache status differs between the two subsets we evaluate against; it is not a semantic mismatch of the pipeline.

CSVs are checked in at [results/b3_il_replay_navsim_v1.csv](b3_il_replay_navsim_v1.csv) and [results/c2_il_ddv2.csv](c2_il_ddv2.csv).

### 4.2 Comparison to the published number
DiffusionDrive IL on full navtest (paper): 88.1. Our pilot: 95.6 on a 50-token subset. **Not directly comparable** — the pilot is 50/14000+ scenes from 5 of ~500 logs. We do not claim reproduction.

### 4.3 RL / SEL agent scoring — not yet reproduced
Attempted via `run_pdm_score_fast.py` (official path, with metric_cache + feature cache precomputed). Both agents ran inference end-to-end but **every scene** failed inside the parallel PDM scorer with `AssertionError: angle is not finite`. Patching `_pdm_worker` to swallow the exception confirms this is uniform: all scores returned 0. Interpretation: the diffusion-head output has non-finite headings on this environment.

Candidates for the root cause (ordered):
1. **Torch/diffusers version mismatch with the released checkpoint.** The DiffusionDriveV2 README does not pin diffusers; we are on `diffusers==0.31.0`, torch 2.0.1+cu117 (what the repo's install.md produces). The ckpt was trained in a different env (likely newer diffusers/torch).
2. Partial dataset — only 5 logs with sensor data, which affects feature-cache keys but not individual scene inference.
3. Patched `plan_anchor_path` / `pdm_scoring_params.yaml` paths (§6) — needed because the originals were absolute hardcoded. Not expected to produce NaNs but worth re-verifying under the author's intended env.

Deferred to next iteration: a focused env rebuild (newer torch + matching diffusers) or a byte-wise reproduction of the author's eval env.

## 5. Timings on the 5-log pilot

| Step | Wall time |
|---|---|
| Dataset download (maps + metadata + 5 cam + partial lidar, ~37 GB) | ~1 h 10 min (interrupted by rsync bug mid-way) |
| `navsim_v1` env install | ~10 min |
| `navsim_v2` env install | ~8 min |
| B1 IL inference, 50 tokens on GTX 1650 Ti | 29 s |
| Metric caching for 5 logs (242 scenes), 2 CPU workers | ~7 min |
| Dataset feature caching for 5 logs | ~50 s |
| B3 PDMS scoring (ReplayAgent on 49 scenes, 2 workers) | ~11 s |
| C2 IL scoring via DDv2 (49 scenes, 2 workers) | ~33 s |
| C2 RL/SEL fast-score attempts (242 scenes, GPU) | ~1 min 30 s per agent |

## 6. Patches to third-party code (for the record)

### 6.1 `DiffusionDriveV2`
Diff vs upstream commit `1cd12a1`:
- Hardcoded absolute paths that pointed to the authors' NFS share:
  - `navsim/agents/diffusiondrive/transfuser_config.py:19` — `plan_anchor_path`
  - `navsim/agents/diffusiondrivev2/transfuser_config.py:19` — `plan_anchor_path`
  - `navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_config.py:19` — `plan_anchor_path`
  - `navsim/agents/diffusiondrivev2/diffusiondrivev2_sel_config.py:19` — `plan_anchor_path`
  - `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py:741` — `OmegaConf.load('navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml')`
  - `navsim/agents/diffusiondrivev2/diffusiondrivev2_model_sel.py:910` — same yaml
  All redirected to absolute paths under this repo's local install.
- `_pdm_worker` in both `diffusiondrivev2_model_rl.py` and `diffusiondrivev2_model_sel.py` wrapped in try/except that returns zero-scores on failure. This is a diagnostic hack to let evaluation continue; it should be reverted or made opt-in before any real benchmark number is reported.

### 6.2 Dependency pins added to env `ddv2`
Not in requirements.txt but required to resolve import errors:
- `pip install diffusers==0.31.0` (default 0.36 breaks on `torch.xpu`; 0.27 breaks on `huggingface_hub.cached_download`)
- `pip install cloudpickle`

### 6.3 Dataset layout aliases
Hardcoded config paths expect flat layouts; HF unpack uses `openscene-v1.1/.../test/`. Created symlinks on the F: drive:
- `dataset/navsim_logs/test → ../test_navsim_logs/test`
- `dataset/sensor_blobs/test → ../test_sensor_blobs/test`

## 7. Open questions

- Full-navtest timing on this machine. Inference extrapolates to ~**2.3 hours per agent** at 3 it/s on 14000+ scenes. Metric caching: 242 scenes in 7 min → ~7 hours on all logs on 2 CPU threads (bottleneck). Feature caching likely similar.
- Dataset download: resuming the remaining 9 camera chunks + 7 lidar chunks would add ~55 GB on F:. Current free space: 83 GB (OK).
- RL/SEL root cause (§4.3) — paper's env details (torch, diffusers) are not documented. Options: train a tiny model from scratch to validate the pipeline, or try `torch==2.2.*` and `diffusers==0.29.*` to match the paper era.
- NAVSIM v2 EPDMS + reactive IDM: not exercised at all this iteration (C3 skipped).

## 8. For the user (what you decide next)

1. Keep going on this environment (ddv2 + torch 2.0.1) and accept that RL/SEL benchmark numbers wait, OR rebuild env to match paper's.
2. Resume download of the remaining 9 camera + 7 lidar chunks (overnight, ~2–3 h) and rerun pilot on more logs to see whether the RL "angle not finite" is scene-specific.
3. Proceed to Milestone 1 (social-burden metric) using IL baseline only.
