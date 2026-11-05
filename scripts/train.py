"""Unified BrainCo training entrypoint, with lazy JAX/PyTorch dispatch.

The legacy JAX helpers remain importable here for existing callers and tests.
Spawned PyTorch data workers can import this module without importing JAX.
"""

import importlib


def __getattr__(name):
    return getattr(importlib.import_module("openpi.training.train_jax"), name)


def main(config):
    from openpi.diffusion.config import DiffusionTrainConfig

    if isinstance(config, DiffusionTrainConfig):
        from openpi.diffusion.train import train

        return train(config)
    return importlib.import_module("openpi.training.train_jax").main(config)


if __name__ == "__main__":
    from openpi.diffusion.cli import maybe_dispatch

    maybe_dispatch()
    from openpi.training.config import cli

    main(cli())
