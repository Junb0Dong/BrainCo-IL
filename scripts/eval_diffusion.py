"""Offline absolute-joint error and latency evaluation; never publishes robot commands."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from openpi.diffusion.checkpoint import load_policy
from openpi.diffusion.config import CAMERAS
from openpi.diffusion.config import GROUPS
from openpi.diffusion.data import DiffusionDataset


def evaluate(checkpoint: str, *, device: str, max_samples: int) -> dict:
    torch.set_num_threads(4)
    started = time.perf_counter()
    cfg, policy = load_policy(checkpoint, device=device)
    load_seconds = time.perf_counter() - started
    dataset = DiffusionDataset(cfg, train=False)
    stride = cfg.model.action_horizon
    selected = range(0, min(len(dataset), max_samples * stride), stride)
    errors, jumps, latencies = [], [], []
    previous, previous_index = None, None
    for index in selected:
        raw_index = int(dataset.indices[index])
        raw = dataset.raw[raw_index]
        observation = {"observation/state": raw["observation.state"][0, cfg.interface["state_indices"]].numpy()}
        for key, target in zip(
            CAMERAS, ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"), strict=True
        ):
            observation[target] = (raw[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        torch.manual_seed(cfg.training.seed + index)
        started = time.perf_counter()
        predicted = policy.infer(observation)["actions"]
        latencies.append((time.perf_counter() - started) * 1000)
        truth = raw["action"][:stride, cfg.interface["action_indices"]].numpy()
        errors.append(np.abs(predicted - truth).mean(axis=0))
        if previous is not None and raw_index == previous_index + stride:
            jumps.append(np.abs(predicted[0] - previous[-1]))
        previous, previous_index = predicted, raw_index
    if not errors:
        raise ValueError("no evaluation samples")
    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "samples": len(errors),
        "load_seconds": load_seconds,
        "cold_infer_ms": latencies[0],
        "warm_infer_p50_ms": float(np.percentile(latencies[1:] or latencies, 50)),
        "warm_infer_p95_ms": float(np.percentile(latencies[1:] or latencies, 95)),
        "chunk_duration_ms": stride / cfg.interface["fps"] * 1000,
        "absolute_joint_mae": {},
        "adjacent_openloop_chunk_jump": {},
        "note": "Recorded-observation evaluation; does not measure closed-loop robot success.",
    }
    mean_errors = np.mean(errors, axis=0)
    mean_jumps = np.mean(jumps, axis=0) if jumps else None
    offset = 0
    for group in GROUPS:
        width = cfg.interface["group_dims"][group]
        result["absolute_joint_mae"][group] = float(mean_errors[offset : offset + width].mean())
        if mean_jumps is not None:
            result["adjacent_openloop_chunk_jump"][group] = float(mean_jumps[offset : offset + width].mean())
        offset += width
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, device=args.device, max_samples=args.max_samples)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
