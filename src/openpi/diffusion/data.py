"""Named 56D selection, shared image processing and train-only delta statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import Dataset

from openpi.diffusion.config import CAMERAS
from openpi.diffusion.config import DiffusionTrainConfig
from openpi.diffusion.config import digest


def arm_mask(cfg: DiffusionTrainConfig) -> np.ndarray:
    return np.asarray(
        [name.startswith(("left_arm_joint", "right_arm_joint")) for name in cfg.interface["action_names"]], dtype=bool
    )


def relative_actions(actions, state, mask):
    """Every target in the chunk is relative to the same current observation."""
    return actions - state[..., None, :] * mask


def absolute_actions(actions, state, mask):
    return actions + state[..., None, :] * mask


def preprocess_image(image, size: int) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("deployment image must be RGB uint8 HWC")
        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255)
    else:
        tensor = torch.as_tensor(image)
        if tensor.ndim != 3 or tensor.shape[0] != 3 or not tensor.is_floating_point():
            raise ValueError("dataset image must be RGB floating CHW")
        if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
            raise ValueError("dataset RGB values must be finite in [0,1]")
        tensor = tensor.float()
    return F.interpolate(tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False, antialias=True)[
        0
    ]


def dataset_metadata(cfg: DiffusionTrainConfig):
    root = Path(cfg.dataset)
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError("this DP backend expects an existing LeRobot v2.1 dataset")
    episodes = {
        int(ep["episode_index"]): ep for ep in map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines())
    }
    selected = set(cfg.train_episodes) | set(cfg.eval_episodes)
    if not selected <= episodes.keys():
        raise ValueError("configured episode IDs are missing from dataset")
    if any(episodes[e]["length"] < cfg.model.horizon for e in selected):
        raise ValueError("episode shorter than prediction horizon")
    for key, semantic in (("observation.state", "state"), ("action", "action")):
        names = info["features"][key]["names"]
        if [names[i] for i in cfg.interface[f"{semantic}_indices"]] != cfg.interface[f"{semantic}_names"]:
            raise ValueError("dataset metadata no longer matches resolved configuration")
    return info, episodes


def numeric_episode(cfg: DiffusionTrainConfig, info, episode: int):
    path = Path(cfg.dataset) / info["data_path"].format(
        episode_chunk=episode // info["chunks_size"], episode_index=episode
    )
    table = pq.read_table(path, columns=["observation.state", "action", "timestamp"])
    arrays = []
    for key, semantic in (("observation.state", "state"), ("action", "action")):
        value = np.asarray(table[key].to_pylist(), dtype=np.float32)
        if value.shape != (len(table), cfg.interface[f"{semantic}_dim"]) or not np.isfinite(value).all():
            raise ValueError(f"invalid numeric data in {path}: {key}")
        arrays.append(value[:, cfg.interface[f"{semantic}_indices"]])
    timestamps = np.asarray(table["timestamp"].to_pylist())
    if not np.isfinite(timestamps).all() or not np.allclose(np.diff(timestamps), 1 / info["fps"], atol=2e-4):
        raise ValueError(f"invalid timestamps in {path}")
    return arrays


def compute_stats(cfg: DiffusionTrainConfig) -> dict:
    """Stream actual transformed training windows, never aggregate absolute action metadata stats."""
    info, episodes = dataset_metadata(cfg)
    accum = {
        key: {
            "min": np.full(cfg.model.action_dim, np.inf),
            "max": np.full(cfg.model.action_dim, -np.inf),
            "sum": np.zeros(cfg.model.action_dim),
            "square": np.zeros(cfg.model.action_dim),
            "count": 0,
        }
        for key in ("observation.state", "action")
    }
    mask = arm_mask(cfg)
    for episode in sorted(set(cfg.train_episodes) | set(cfg.eval_episodes)):
        state, action = numeric_episode(cfg, info, episode)
        if len(state) != episodes[episode]["length"]:
            raise ValueError(f"episode {episode} metadata/frame count mismatch")
        if episode not in cfg.train_episodes:
            continue
        count = len(state) - cfg.model.horizon + 1
        windows = np.moveaxis(np.lib.stride_tricks.sliding_window_view(action, cfg.model.horizon, axis=0), -1, 1)
        for start in range(0, count, 256):
            states = state[start : min(start + 256, count)]
            targets = relative_actions(windows[start : start + len(states)], states, mask)
            for key, raw_values in (("observation.state", states), ("action", targets)):
                values = raw_values.reshape(-1, cfg.model.action_dim).astype(np.float64)
                a = accum[key]
                a["min"] = np.minimum(a["min"], values.min(axis=0))
                a["max"] = np.maximum(a["max"], values.max(axis=0))
                a["sum"] += values.sum(axis=0)
                a["square"] += np.square(values).sum(axis=0)
                a["count"] += len(values)
    stats = {}
    for key, a in accum.items():
        mean = a["sum"] / a["count"]
        stats[key] = {
            "min": a["min"].tolist(),
            "max": a["max"].tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(a["square"] / a["count"] - mean**2, 0)).tolist(),
        }
    for camera in CAMERAS:
        stats[camera] = {
            "mean": np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1).tolist(),
            "std": np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1).tolist(),
        }
    return {
        "schema_version": 1,
        "signature": stats_signature(cfg),
        "stats": stats,
        "train_episodes": list(cfg.train_episodes),
        "eval_episodes": list(cfg.eval_episodes),
        "action_encoding": "arms_relative_to_current_state_hands_absolute",
        "state_count": accum["observation.state"]["count"],
        "action_count": accum["action"]["count"],
    }


def stats_signature(cfg: DiffusionTrainConfig) -> str:
    return digest(
        {
            "interface": cfg.interface,
            "horizon": cfg.model.horizon,
            "train_episodes": cfg.train_episodes,
            "image_size": cfg.image_size,
            "encoding": "arms_relative_to_current_state_hands_absolute",
        }
    )


def tensor_stats(payload: dict):
    return {
        key: {name: torch.tensor(value, dtype=torch.float32) for name, value in entry.items()}
        for key, entry in payload["stats"].items()
    }


def _patch_legacy_columns():
    # The fixed LeRobot version predates datasets.Column. Keep the compatibility
    # shim local to the old loader and do not import the JAX training pipeline.
    if getattr(torch.stack, "_openpi_datasets_column_compat", False):
        return
    original = torch.stack

    def stack(values, *args, **kwargs):
        if values.__class__.__name__ == "Column" and values.__class__.__module__.startswith("datasets."):
            values = list(values)
            if not values or not torch.is_tensor(values[0]):
                return torch.as_tensor(values)
        return original(values, *args, **kwargs)

    stack._openpi_datasets_column_compat = True  # noqa: SLF001
    torch.stack = stack


class DiffusionDataset(Dataset):
    def __init__(self, cfg: DiffusionTrainConfig, *, train: bool):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.cfg = cfg
        dataset_metadata(cfg)
        _patch_legacy_columns()
        # Load all episodes: this old LeRobot indexes boundaries by global episode ID.
        # The wrapper selects train/eval windows without changing those IDs.
        self.raw = LeRobotDataset(
            repo_id=cfg.dataset,
            root=cfg.dataset,
            video_backend="pyav",
            tolerance_s=2e-4,
            delta_timestamps={
                "observation.state": [0.0],
                "action": [i / cfg.interface["fps"] for i in range(cfg.model.horizon)],
            },
        )
        episodes = cfg.train_episodes if train else cfg.eval_episodes
        bounds = self.raw.episode_data_index
        self.indices = np.concatenate(
            [np.arange(int(bounds["from"][e]), int(bounds["to"][e]) - cfg.model.horizon + 1) for e in episodes]
        )
        self.mask = torch.from_numpy(arm_mask(cfg))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        # Spawned workers have their own torch module; patches in __init__ run
        # only in the parent and are not serialized with the dataset object.
        _patch_legacy_columns()
        raw = self.raw[int(self.indices[index])]
        state = raw["observation.state"][..., self.cfg.interface["state_indices"]].float().reshape(1, -1)
        action = raw["action"][..., self.cfg.interface["action_indices"]].float()
        if raw["action_is_pad"].any():
            raise ValueError("complete-window sampler unexpectedly encountered padded actions")
        result = {
            "observation.state": state,
            "action": relative_actions(action, state[0], self.mask),
            "action_is_pad": raw["action_is_pad"],
        }
        for key in CAMERAS:
            result[key] = preprocess_image(raw[key], self.cfg.image_size).unsqueeze(0)
        return result
