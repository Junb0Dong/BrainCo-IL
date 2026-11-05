"""Single-GPU DP training with deterministic episode sampling and resumable EMA."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import Subset

from openpi.diffusion.checkpoint import save_checkpoint
from openpi.diffusion.checkpoint import update_alias
from openpi.diffusion.checkpoint import verify_artifact
from openpi.diffusion.config import DiffusionTrainConfig
from openpi.diffusion.config import config_yaml
from openpi.diffusion.config import digest
from openpi.diffusion.data import DiffusionDataset
from openpi.diffusion.data import compute_stats
from openpi.diffusion.policy import create_model


class WindowBatchSampler(Sampler):
    """Sample position derives from completed optimizer steps, independent of prefetch."""

    def __init__(self, size: int, seed: int, start: int, stop: int, batch_size: int):
        self.size, self.seed, self.start, self.stop, self.batch_size = size, seed, start, stop, batch_size

    def __iter__(self):
        epoch = -1
        batch = []
        for position in range(self.start, self.stop):
            next_epoch, index = divmod(position, self.size)
            if next_epoch != epoch:
                epoch = next_epoch
                order = np.random.default_rng(self.seed + epoch).permutation(self.size)
            batch.append(int(order[index]))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def __len__(self):
        return math.ceil((self.stop - self.start) / self.batch_size)


def worker_init(_worker_id):
    torch.set_num_threads(1)


def autocast(cfg):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if (cfg.training.precision == "bf16" and cfg.training.device.startswith("cuda"))
        else contextlib.nullcontext()
    )


def device_batch(batch: dict, device: str) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_loader(dataset, cfg, *, batch_sampler=None, batch_size=None):
    options = {
        "num_workers": cfg.training.num_workers,
        "pin_memory": cfg.training.device.startswith("cuda"),
        "worker_init_fn": worker_init,
        "generator": torch.Generator().manual_seed(cfg.training.seed),
    }
    if cfg.training.num_workers:
        options.update(multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=2)
    if batch_sampler is not None:
        return DataLoader(dataset, batch_sampler=batch_sampler, **options)
    return DataLoader(dataset, batch_size=batch_size or cfg.training.microbatch_size, **options)


@torch.no_grad()
def evaluate(model, loader, cfg) -> float:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    devices = (
        [torch.device(cfg.training.device).index or torch.cuda.current_device()]
        if (cfg.training.device.startswith("cuda"))
        else []
    )
    # Evaluation has fixed noise and never advances the training RNG stream.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(cfg.training.seed + 100000)
        for raw in loader:
            batch = device_batch(raw, cfg.training.device)
            with autocast(cfg):
                loss, _ = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite held-out loss")
            size = len(batch["action"])
            total += float(loss) * size
            count += size
    model.train(was_training)
    return total / count


def rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def write_status(directory, payload):
    path = directory / ".status.json.tmp"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    path.replace(directory / "status.json")


def train(cfg: DiffusionTrainConfig, *, resume: str | None = None) -> None:
    cfg.validate()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.set_num_threads(4)
    t = cfg.training
    random.seed(t.seed)
    np.random.seed(t.seed)
    torch.manual_seed(t.seed)
    if t.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(t.device).index or 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run = cfg.checkpoint_dir
    if (run / "status.json").exists() and not resume:
        raise FileExistsError(f"run exists; supply --resume or a new --output-dir: {run}")
    run.mkdir(parents=True, exist_ok=True)
    state = None
    if resume:
        source = Path(resume).expanduser().resolve()
        previous = verify_artifact(source)
        for key in ("interface", "model", "image_size", "train_episodes", "eval_episodes"):
            old, new = getattr(previous, key), getattr(cfg, key)
            if dataclasses.is_dataclass(old):
                old, new = dataclasses.asdict(old), dataclasses.asdict(new)
            if digest(old) != digest(new):
                raise ValueError(f"resume cannot change {key}")
        # Only locally produced, explicitly selected training state is unpickled.
        state = torch.load(source / "train_state.pt", map_location="cpu", weights_only=False)
        stats = json.loads((source / "assets/diffusion_stats.json").read_text())
    else:
        logging.info("Validating numeric episodes and computing train-only delta statistics")
        stats = compute_stats(cfg)
    (run / "diffusion_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (run / "train_config.yaml").write_text(config_yaml(cfg, cfg.name))
    model = create_model(cfg, stats).train()
    ema = copy.deepcopy(model).eval().requires_grad_(requires_grad=False)
    optimizer = torch.optim.Adam(
        model.get_optim_params(), lr=t.peak_lr, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6
    )

    def lr_multiplier(step):
        if step < t.warmup_steps:
            return (step + 1) / max(1, t.warmup_steps)
        progress = min(1.0, (step - t.warmup_steps) / max(1, t.num_steps - t.warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    start, best, best_step = 0, float("inf"), None
    if state is not None:
        model.load_state_dict(state["model"], strict=True)
        ema.load_state_dict(state["ema"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best, best_step = state["step"], state["best_loss"], state["best_step"]
        if previous.training.max_eval_samples != t.max_eval_samples:
            # Scores from different held-out subsets are not comparable.
            best, best_step = float("inf"), None
        if start >= t.num_steps:
            raise ValueError("resume checkpoint has already reached requested num_steps")
    training_data = DiffusionDataset(cfg, train=True)
    evaluation_data = DiffusionDataset(cfg, train=False)
    eval_indices = (
        np.random.default_rng(t.seed)
        .choice(len(evaluation_data), min(t.max_eval_samples, len(evaluation_data)), replace=False)
        .tolist()
    )
    eval_loader = make_loader(Subset(evaluation_data, eval_indices), cfg)
    sampler = WindowBatchSampler(
        len(training_data), t.seed, start * t.batch_size, t.num_steps * t.batch_size, t.microbatch_size
    )
    loader = make_loader(training_data, cfg, batch_sampler=sampler)
    batches = iter(loader)
    if state is not None:
        restore_rng(state["rng"])
        del state
    else:
        logging.info("Initial held-out loss: %.6f", evaluate(ema, eval_loader, cfg))
    started = time.perf_counter()
    window_started = started
    losses = []
    last_checkpoint = None
    write_status(run, {"state": "running", "step": start, "target_steps": t.num_steps, "resume_from": resume})
    try:
        for step in range(start + 1, t.num_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            accumulation = t.batch_size // t.microbatch_size
            for _ in range(accumulation):
                batch = device_batch(next(batches), t.device)
                with autocast(cfg):
                    loss, _ = model(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                (loss / accumulation).backward()
                loss_value += float(loss.detach()) / accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for target, source in zip(ema.diffusion.parameters(), model.diffusion.parameters(), strict=True):
                    target.lerp_(source, 1 - t.ema_decay)
                for target, source in zip(ema.diffusion.buffers(), model.diffusion.buffers(), strict=True):
                    target.copy_(source)
            losses.append(loss_value)
            metrics = None
            improved = False
            if step % t.eval_interval == 0 or step == t.num_steps:
                live_loss = evaluate(model, eval_loader, cfg)
                ema_loss = evaluate(ema, eval_loader, cfg)
                improved = ema_loss < best
                if improved:
                    best, best_step = ema_loss, step
                metrics = {"step": step, "eval_loss_live": live_loss, "eval_loss_ema": ema_loss}
                logging.info("step=%d held-out live=%.6f EMA=%.6f", step, live_loss, ema_loss)
            if step % t.log_interval == 0 or step == t.num_steps:
                seconds = time.perf_counter() - window_started
                report = {
                    "step": step,
                    "loss": float(np.mean(losses)),
                    "grad_norm": float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                    "steps_per_second": len(losses) / seconds,
                    "elapsed_seconds": time.perf_counter() - started,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated() if t.device.startswith("cuda") else 0,
                    **(metrics or {}),
                }
                with (run / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(report) + "\n")
                logging.info("step=%d loss=%.6f steps/s=%.3f", step, report["loss"], report["steps_per_second"])
                write_status(run, {"state": "running", "target_steps": t.num_steps, **report})
                losses.clear()
                window_started = time.perf_counter()
            elif metrics:
                with (run / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(metrics) + "\n")
            if improved or step % t.save_interval == 0 or step == t.num_steps:
                training_state = {
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": rng_state(),
                    "best_loss": best,
                    "best_step": best_step,
                }
                last_checkpoint = save_checkpoint(cfg, ema, stats, step, training_state=training_state)
                if improved:
                    update_alias(run, "best", str(step))
                logging.info("Saved %s", last_checkpoint)
        write_status(
            run,
            {
                "state": "complete",
                "step": t.num_steps,
                "best_step": best_step,
                "best_eval_loss_ema": best,
                "last_checkpoint": str(last_checkpoint),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    except BaseException as exc:
        write_status(run, {"state": "failed", "error": repr(exc), "last_checkpoint": str(last_checkpoint)})
        raise
