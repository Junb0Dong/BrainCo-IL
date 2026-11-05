"""Dispatch DP commands before the legacy JAX entrypoints are imported."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import yaml

from openpi.diffusion.config import is_diffusion_payload
from openpi.diffusion.config import load_config


def maybe_dispatch(*, stats_only: bool = False) -> None:
    args = sys.argv[1:]
    candidate = args[args.index("--config-path") + 1] if "--config-path" in args else (args[0] if args else "")
    path = Path(candidate)
    if not path.is_file() or path.suffix not in (".yaml", ".yml"):
        return
    if not is_diffusion_payload(yaml.safe_load(path.read_text())):
        return
    parser = argparse.ArgumentParser(description="BrainCo-IL PyTorch Diffusion Policy")
    parser.add_argument("config", nargs="?")
    parser.add_argument("--config-path")
    parser.add_argument("--resume", help="Completed step directory containing train_state.pt")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir", help="Run directory (must be new unless resuming)")
    parser.add_argument("--device")
    parsed = parser.parse_args(args)
    cfg = load_config(parsed.config_path or parsed.config)
    overrides = {}
    if parsed.steps is not None:
        overrides["num_steps"] = parsed.steps
    if parsed.device:
        overrides["device"] = parsed.device
    if parsed.output_dir:
        directory = Path(parsed.output_dir).expanduser().resolve()
        overrides["checkpoint_base_dir"] = str(directory.parent)
        cfg = dataclasses.replace(cfg, name=directory.name)
    cfg = dataclasses.replace(cfg, training=dataclasses.replace(cfg.training, **overrides))
    cfg.validate()
    if stats_only:
        from openpi.diffusion.data import compute_stats

        target = cfg.checkpoint_dir / "diffusion_stats.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(compute_stats(cfg), indent=2) + "\n")
        print(target)
    else:
        from openpi.diffusion.train import train

        train(cfg, resume=parsed.resume)
    raise SystemExit(0)
