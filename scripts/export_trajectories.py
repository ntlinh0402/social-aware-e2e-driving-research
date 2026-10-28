#!/usr/bin/env python3
"""Export trajectories from DiffusionDriveV2 agents into the handoff format.

Produces for each run:
    handoff/<run_id>/
        trajectories.pkl   # dict: token -> {"poses": np.ndarray (N, 3)}
        tokens.txt
        manifest.json

Usage (inside env ddv2, with NAVSIM_DEVKIT_ROOT and OPENSCENE_DATA_ROOT set):
    python scripts/export_trajectories.py \
        --agent rl \
        --ckpt /home/linh/navsim_workspace/DiffusionDriveV2/ckpts/diffusiondrivev2_rl.ckpt \
        --split navtest \
        --max-scenes 50 \
        --out /mnt/o/research/handoff \
        --device cuda

Agent choices:
    il   -> diffusiondrive_agent.yaml   (TransfuserAgent, baseline IL)
    rl   -> diffusiondrivev2_rl_agent.yaml
    sel  -> diffusiondrivev2_sel_agent.yaml
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import pickle
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from tqdm import tqdm

logger = logging.getLogger(__name__)

AGENT_CONFIG = {
    "il":  "diffusiondrive_agent",
    "rl":  "diffusiondrivev2_rl_agent",
    "sel": "diffusiondrivev2_sel_agent",
}
AGENT_MODEL_NAME = {
    "il":  "diffusiondrive_il",
    "rl":  "diffusiondrivev2_rl",
    "sel": "diffusiondrivev2_sel",
}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def git_head(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def build_agent_and_loader(agent_key: str, ckpt: Path, split: str, max_scenes: int, config_dir: Path):
    """Use hydra to compose the standard submission config, then override agent + scene filter."""
    from navsim.common.dataloader import SceneLoader  # noqa: F401 (ensures navsim import path works)

    # Hydra config dir sits inside the DDv2 repo
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        overrides = [
            f"agent={AGENT_CONFIG[agent_key]}",
            f"agent.checkpoint_path={ckpt}",
            f"train_test_split=navtest",
            f"train_test_split.scene_filter.max_scenes={max_scenes}" if max_scenes else "",
            # Fill required fields of default_run_create_submission_pickle (unused by us)
            'team_name=research',
            'authors=["linh"]',
            'email="nguyenthuylinh040205@gmail.com"',
            'institution="VNU/UET"',
            'country="VN"',
        ]
        overrides = [o for o in overrides if o]
        cfg = compose(config_name="default_run_create_submission_pickle", overrides=overrides)

    agent = instantiate(cfg.agent)
    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    return agent, scene_filter, cfg


def to_device(features, device):
    out = {}
    for k, v in features.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_inference_loop(agent, scene_loader, device: str) -> dict:
    """Replicates abstract_agent.compute_trajectory but runs model on the requested device."""
    agent = agent.to(device)
    agent.eval()
    output = {}
    for token in tqdm(scene_loader, desc="inference"):
        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            features = {}
            for builder in agent.get_feature_builders():
                features.update(builder.compute_features(agent_input))
            features = {k: v.unsqueeze(0) for k, v in features.items()}
            features = to_device(features, device)
            with torch.no_grad():
                predictions = agent.forward(features)
            poses = predictions["trajectory"].squeeze(0).detach().cpu().numpy().astype(np.float32)
            output[token] = {"poses": poses}
        except Exception:
            logger.warning("agent failed on token %s", token)
            traceback.print_exc()
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=list(AGENT_CONFIG.keys()))
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--max-scenes", type=int, default=50, help="0 = all")
    ap.add_argument("--subset-label", default=None, help="label used in run_id; default auto")
    ap.add_argument("--out", required=True, type=Path, help="base handoff dir (research/handoff)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    devkit = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    config_dir = devkit / "navsim" / "planning" / "script" / "config" / "pdm_scoring"
    if not config_dir.is_dir():
        sys.exit(f"config dir not found: {config_dir}")

    data_path = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "test_navsim_logs"
    sensor_blobs = Path(os.environ["OPENSCENE_DATA_ROOT"]) / "test_sensor_blobs"

    # Build agent + scene filter via hydra
    agent, scene_filter, cfg = build_agent_and_loader(
        args.agent, args.ckpt.resolve(), args.split, args.max_scenes, config_dir,
    )
    agent.initialize()  # load checkpoint

    # Build scene loader
    from navsim.common.dataloader import SceneLoader
    scene_loader = SceneLoader(
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_blobs_path=sensor_blobs,
        sensor_config=agent.get_sensor_config(),
    )

    # Run inference
    output = run_inference_loop(agent, scene_loader, args.device)

    # Build run_id
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    subset = args.subset_label or (f"pilot_n{args.max_scenes}" if args.max_scenes else "all")
    run_id = f"{AGENT_MODEL_NAME[args.agent]}_{args.split}_{subset}_{stamp}"
    out_dir = args.out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "trajectories.pkl", "wb") as f:
        pickle.dump(output, f)
    (out_dir / "tokens.txt").write_text("\n".join(sorted(output.keys())) + "\n")

    # Trajectory sampling spec taken from the agent's config (not guessed)
    ts = cfg.agent.config.trajectory_sampling
    manifest = {
        "run_id": run_id,
        "model": AGENT_MODEL_NAME[args.agent],
        "checkpoint_file": str(args.ckpt),
        "checkpoint_sha256": sha256_of(args.ckpt),
        "split": args.split,
        "subset": subset,
        "num_tokens": len(output),
        "trajectory_sampling": {
            "num_poses": int(ts.time_horizon / ts.interval_length),
            "time_horizon": float(ts.time_horizon),
            "interval_length": float(ts.interval_length),
        },
        "repo_commits": {
            "DiffusionDriveV2": git_head(devkit),
            "research": git_head(Path(__file__).resolve().parents[1]),
        },
        "device": args.device,
        "batch_size": 1,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWROTE {out_dir}")
    print(f"  tokens: {len(output)}")
    print(f"  manifest: {manifest['run_id']}")


if __name__ == "__main__":
    main()
