"""Lightweight experiment and immutable runtime configuration for LeRobot DP."""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from openpi.deploy.contract import resolve_dataset_interface

GROUPS = ("left_arm", "right_arm", "left_hand", "right_hand")
CAMERAS = (
    "observation.images.cam_head",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
RUNTIME = "lerobot_diffusion"


class ModelType(enum.StrEnum):
    DIFFUSION = "diffusion"


@dataclasses.dataclass(frozen=True)
class DiffusionModelConfig:
    action_dim: int = 56
    action_horizon: int = 16
    horizon: int = 32
    n_obs_steps: int = 1
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    num_train_timesteps: int = 100
    num_inference_steps: int = 20
    noise_scheduler_type: str = "DDIM"

    @property
    def model_type(self) -> ModelType:
        return ModelType.DIFFUSION


@dataclasses.dataclass(frozen=True)
class DiffusionTrainingConfig:
    seed: int = 42
    batch_size: int = 16
    microbatch_size: int = 4
    num_workers: int = 4
    num_steps: int = 16000
    peak_lr: float = 1e-4
    warmup_steps: int = 500
    ema_decay: float = 0.999
    save_interval: int = 2000
    eval_interval: int = 1000
    log_interval: int = 50
    max_eval_samples: int = 2048
    precision: str = "bf16"
    device: str = "cuda"
    checkpoint_base_dir: str = "./checkpoints/diffusion"


@dataclasses.dataclass(frozen=True)
class DiffusionTrainConfig:
    name: str
    dataset: str
    interface: dict[str, Any]
    model: DiffusionModelConfig = dataclasses.field(default_factory=DiffusionModelConfig)
    training: DiffusionTrainingConfig = dataclasses.field(default_factory=DiffusionTrainingConfig)
    image_size: int = 256
    train_episodes: tuple[int, ...] = tuple(range(45))
    eval_episodes: tuple[int, ...] = tuple(range(45, 50))

    @property
    def exp_name(self) -> str:
        return self.name

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.training.checkpoint_base_dir).expanduser().resolve() / self.name

    def validate(self) -> None:
        m, t = self.model, self.training
        if not self.name or Path(self.name).name != self.name or self.name in (".", ".."):
            raise ValueError("experiment.name must be a single directory name")
        if m.n_obs_steps != 1:
            raise ValueError("Diffusion protocol v1 supports n_obs_steps=1 only")
        if not m.down_dims or any(d <= 0 or d % 8 for d in m.down_dims):
            raise ValueError("down_dims must contain positive multiples of 8")
        if m.horizon <= 0 or m.horizon % (2 ** len(m.down_dims)):
            raise ValueError("horizon must be a positive multiple of the U-Net downsampling factor")
        if not 1 <= m.action_horizon <= m.horizon:
            raise ValueError("action_horizon must be in [1, horizon]")
        if m.noise_scheduler_type not in ("DDIM", "DDPM"):
            raise ValueError("noise_scheduler_type must be DDIM or DDPM")
        if not 1 <= m.num_inference_steps <= m.num_train_timesteps:
            raise ValueError("invalid diffusion timestep counts")
        names = self.interface["state_names"]
        if names != self.interface["action_names"] or len(names) != m.action_dim:
            raise ValueError("state/action names and model dimensions differ")
        if self.interface["group_dims"] != dict(zip(GROUPS, (7, 7, 21, 21), strict=True)):
            raise ValueError("first DP release requires the 56D Revo3 interface")
        if len(names) != len(set(names)) or tuple(self.interface["cameras"]) != CAMERAS:
            raise ValueError("invalid joint names or camera order")
        if self.interface["fps"] <= 0 or self.image_size < 32:
            raise ValueError("invalid FPS or image_size")
        for key in ("state", "action"):
            indices = self.interface[f"{key}_indices"]
            if len(indices) != m.action_dim or len(set(indices)) != len(indices):
                raise ValueError(f"invalid {key} selection")
            if min(indices) < 0 or max(indices) >= self.interface[f"{key}_dim"]:
                raise ValueError(f"{key} indices out of bounds")
        if not self.train_episodes or not self.eval_episodes or set(self.train_episodes) & set(self.eval_episodes):
            raise ValueError("train/eval episodes must be nonempty and disjoint")
        for episodes in (self.train_episodes, self.eval_episodes):
            if min(episodes) < 0 or len(set(episodes)) != len(episodes):
                raise ValueError("episode IDs must be unique and nonnegative")
        for key in (
            "batch_size",
            "microbatch_size",
            "num_steps",
            "save_interval",
            "eval_interval",
            "log_interval",
            "max_eval_samples",
        ):
            if getattr(t, key) <= 0:
                raise ValueError(f"training.{key} must be positive")
        if t.batch_size % t.microbatch_size or t.num_workers < 0 or t.warmup_steps < 0:
            raise ValueError("invalid batch, worker or warmup configuration")
        if not 0 <= t.ema_decay < 1 or t.peak_lr <= 0 or t.precision not in ("bf16", "fp32"):
            raise ValueError("invalid optimizer, EMA or precision configuration")


def digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def is_diffusion_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and (
        payload.get("base") == "diffusion" or payload.get("runtime", {}).get("kind") == RUNTIME
    )


def load_config(path_or_dir: str | Path) -> DiffusionTrainConfig:
    path = Path(path_or_dir).expanduser().resolve()
    if path.is_dir():
        path /= "train_config.yaml"
    payload = yaml.safe_load(path.read_text())
    if not is_diffusion_payload(payload) or payload.get("schema_version") != 1:
        raise ValueError("expected a schema_version: 1 diffusion config")
    if "base" in payload:
        allowed = {"schema_version", "base", "experiment", "policy", "model", "training", "data"}
        if set(payload) - allowed:
            raise ValueError(f"unknown experiment fields: {set(payload) - allowed}")
        exp = dict(payload["experiment"])
        if set(exp) - {"name", "dataset"}:
            raise ValueError("diffusion experiment accepts only name and dataset")
        dataset = Path(exp["dataset"]).expanduser().resolve()
        policy = dict(payload.get("policy", {}))
        if set(policy) - {"groups", "action_horizon"} or tuple(policy.get("groups", GROUPS)) != GROUPS:
            raise ValueError("DP requires canonical groups; action representation is internal")
        interface = resolve_dataset_interface([dataset], GROUPS)
        interface["cameras"] = list(CAMERAS) if set(interface["cameras"]) == set(CAMERAS) else interface["cameras"]
        model = dict(payload.get("model", {}))
        if set(model) & {"action_dim", "action_horizon"}:
            raise ValueError("action_dim is resolved; set action_horizon under policy")
        cfg = DiffusionTrainConfig(
            name=str(exp["name"]),
            dataset=str(dataset),
            interface=interface,
            model=DiffusionModelConfig(
                action_dim=len(interface["action_names"]), action_horizon=int(policy.get("action_horizon", 16)), **model
            ),
            training=DiffusionTrainingConfig(**payload.get("training", {})),
            **payload.get("data", {}),
        )
    else:
        if payload.get("generated") is not True or payload.get("do_not_edit") is not True:
            raise ValueError("runtime config must be generated and immutable")
        if payload.get("generated_by") != "BrainCo-IL" or not payload.get("checkpoint_id"):
            raise ValueError("invalid runtime config header")
        signed = {"runtime": payload["runtime"], "train_config": payload["train_config"]}
        if payload.get("content_sha256") != digest(signed):
            raise ValueError("generated train_config.yaml hash mismatch")
        if payload["runtime"] != {"kind": RUNTIME, "weights": "params", "normalization": "assets/diffusion_stats.json"}:
            raise ValueError("unsupported DP runtime paths")
        fields = dict(payload["train_config"])
        fields["model"] = DiffusionModelConfig(**fields["model"])
        fields["training"] = DiffusionTrainingConfig(**fields["training"])
        cfg = DiffusionTrainConfig(**fields)
    cfg.validate()
    return cfg


def config_yaml(cfg: DiffusionTrainConfig, checkpoint_id: str) -> str:
    # JSON round trip also makes tuples portable without Python-specific YAML tags.
    fields = json.loads(json.dumps(dataclasses.asdict(cfg)))
    signed = {
        "runtime": {"kind": RUNTIME, "weights": "params", "normalization": "assets/diffusion_stats.json"},
        "train_config": fields,
    }
    return yaml.safe_dump(
        {
            "schema_version": 1,
            "generated": True,
            "do_not_edit": True,
            "generated_by": "BrainCo-IL",
            "checkpoint_id": checkpoint_id,
            "content_sha256": digest(signed),
            **signed,
        },
        sort_keys=False,
    )
