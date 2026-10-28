"""Replay agent that returns pre-computed trajectories from a handoff pickle.

Works with both NAVSIM v1.1 (navsim_v1 env) and v2 (navsim_v2 env), since
`AbstractAgent` has the same contract in both.

Expected handoff layout:
    <handoff_dir>/
        trajectories.pkl   # dict: token -> {"poses": np.ndarray (N, 3)}
        tokens.txt
        manifest.json

Usage from a hydra agent config:
    agent:
      _target_: scripts.replay_agent.ReplayAgent
      handoff_dir: /path/to/handoff/<run_id>
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory


class ReplayAgent(AbstractAgent):
    """Looks up a token's trajectory from a precomputed handoff and returns it."""

    def __init__(self, handoff_dir: str):
        # requires_scene=True so the evaluator passes a Scene, from which we get the token.
        super().__init__(requires_scene=True)
        self._handoff_dir = Path(handoff_dir)
        self._traj: Dict[str, Dict[str, Any]] = {}
        self._manifest: Dict[str, Any] = {}

    def name(self) -> str:
        model = self._manifest.get("model") or self._handoff_dir.name
        return f"replay::{model}"

    def get_sensor_config(self) -> SensorConfig:
        # No sensors needed, we only need the token to look up the trajectory.
        return SensorConfig.build_no_sensors()

    def initialize(self) -> None:
        traj_path = self._handoff_dir / "trajectories.pkl"
        manifest_path = self._handoff_dir / "manifest.json"
        with open(traj_path, "rb") as f:
            self._traj = pickle.load(f)
        if manifest_path.is_file():
            with open(manifest_path) as f:
                self._manifest = json.load(f)

    def _lookup_poses(self, token: str) -> np.ndarray:
        if token not in self._traj:
            raise KeyError(f"token {token!r} not in handoff {self._handoff_dir}")
        return np.asarray(self._traj[token]["poses"], dtype=np.float32)

    def compute_trajectory(self, agent_input: AgentInput, scene=None) -> Trajectory:  # type: ignore[override]
        if scene is None:
            raise RuntimeError("ReplayAgent requires_scene=True; evaluator must pass scene.")
        token = scene.scene_metadata.initial_token
        poses = self._lookup_poses(token)
        return Trajectory(poses)

    # No feature builders / forward — the agent short-circuits inference.
    def get_feature_builders(self) -> List[Any]:
        return []
