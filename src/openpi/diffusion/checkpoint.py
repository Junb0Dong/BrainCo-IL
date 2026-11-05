"""Atomic, self-contained DP artifacts with a checkpoint-v1 deployment envelope."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import yaml

from openpi.deploy.contract import load_policy_contract
from openpi.diffusion.config import GROUPS
from openpi.diffusion.config import DiffusionTrainConfig
from openpi.diffusion.config import config_yaml
from openpi.diffusion.config import digest
from openpi.diffusion.config import load_config


def build_contract(cfg: DiffusionTrainConfig, checkpoint_id: str) -> dict:
    groups, outputs = {}, {}
    offset = 0
    for group in GROUPS:
        width = cfg.interface["group_dims"][group]
        groups[group] = cfg.interface["state_names"][offset : offset + width]
        outputs[group] = cfg.interface["action_names"][offset : offset + width]
        offset += width
    contract = {
        "generated": True,
        "do_not_edit": True,
        "generated_by": "BrainCo-IL",
        "protocol_version": 1,
        "checkpoint_id": checkpoint_id,
        "robot_interface": cfg.interface["robot_type"],
        "required_joint_groups": groups,
        "required_cameras": cfg.interface["cameras"],
        "output_joint_groups": outputs,
        "chunk_steps": cfg.model.action_horizon,
        "policy_rate_hz": cfg.interface["fps"],
        "inference_strategies": ["standard"],
    }
    return {**contract, "contract_hash": digest(contract)}


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_artifact(root: str | Path) -> DiffusionTrainConfig:
    root = Path(root)
    cfg = load_config(root)
    header = yaml.safe_load((root / "train_config.yaml").read_text())
    contract = load_policy_contract(root)
    if contract != build_contract(cfg, header["checkpoint_id"]):
        raise ValueError("DP contract does not match runtime configuration/checkpoint ID")
    manifest = json.loads((root / "inference_manifest.json").read_text())
    expected = {
        "params/config.json",
        "params/model.safetensors",
        "assets/diffusion_stats.json",
        "train_config.yaml",
        "policy_contract.json",
    }
    if set(manifest) != expected:
        raise ValueError("invalid DP inference manifest")
    inference_files = {
        str(path.relative_to(root))
        for directory in ("params", "assets")
        for path in (root / directory).rglob("*")
        if path.is_file()
    }
    if inference_files != {name for name in expected if name.startswith(("params/", "assets/"))}:
        raise ValueError("missing or unexpected files in DP inference assets")
    for name, checksum in manifest.items():
        path = root / name
        if not path.is_file() or file_sha256(path) != checksum:
            raise ValueError(f"missing or modified inference asset: {name}")
    return cfg


def save_checkpoint(cfg, model, stats, step: int, *, training_state: dict | None = None) -> Path:
    import torch

    final = cfg.checkpoint_dir / str(step)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(f"refusing to overwrite completed checkpoint {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{step}-", dir=final.parent))
    try:
        model.save_pretrained(staging / "params")
        (staging / "assets").mkdir()
        (staging / "assets/diffusion_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
        checkpoint_id = f"{cfg.name}_step{step}"
        (staging / "train_config.yaml").write_text(config_yaml(cfg, checkpoint_id))
        (staging / "policy_contract.json").write_text(json.dumps(build_contract(cfg, checkpoint_id), indent=2) + "\n")
        names = [
            "params/config.json",
            "params/model.safetensors",
            "assets/diffusion_stats.json",
            "train_config.yaml",
            "policy_contract.json",
        ]
        (staging / "inference_manifest.json").write_text(
            json.dumps({name: file_sha256(staging / name) for name in names}, indent=2) + "\n"
        )
        if training_state is not None:
            torch.save(training_state, staging / "train_state.pt")
        verify_artifact(staging)
        os.rename(staging, final)
    except BaseException:
        shutil.rmtree(staging)
        raise
    update_alias(final.parent, "last", final.name)
    return final


def update_alias(directory: Path, name: str, target: str) -> None:
    temporary = directory / f".{name}-{os.getpid()}"
    temporary.symlink_to(target, target_is_directory=True)
    temporary.replace(directory / name)


def load_policy(root: str | Path, *, device: str | None = None):
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.configs.policies import PreTrainedConfig
    import torch

    from openpi.diffusion.data import stats_signature
    from openpi.diffusion.policy import BrainCoDiffusionPolicy
    from openpi.diffusion.policy import model_config

    root = Path(root)
    cfg = verify_artifact(root)
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    saved = PreTrainedConfig.from_pretrained(root / "params")
    expected = model_config(cfg, device=target)
    actual_fields, expected_fields = dataclasses.asdict(saved), dataclasses.asdict(expected)
    actual_fields.pop("device", None)
    expected_fields.pop("device", None)
    if digest(actual_fields) != digest(expected_fields):
        raise ValueError("saved LeRobot config differs from resolved BrainCo config")
    stats = json.loads((root / "assets/diffusion_stats.json").read_text())
    if stats["signature"] != stats_signature(cfg):
        raise ValueError("normalization statistics belong to a different pipeline")
    model = (
        DiffusionPolicy.from_pretrained(root / "params", config=expected, strict=True, local_files_only=True)
        .to(target)
        .eval()
    )
    # Normalizers are inference dependencies inside model.safetensors. Verify the
    # readable statistics are the same values, including constant joint ranges.
    for key, entry in stats["stats"].items():
        module = model.normalize_targets if key == "action" else model.normalize_inputs
        buffer = getattr(module, "buffer_" + key.replace(".", "_"))
        for name, tensor in buffer.items():
            reference = torch.tensor(entry[name], dtype=tensor.dtype, device=tensor.device)
            if not torch.equal(tensor, reference):
                raise ValueError(f"normalizer mismatch: {key}.{name}")
    return cfg, BrainCoDiffusionPolicy(cfg, model)
