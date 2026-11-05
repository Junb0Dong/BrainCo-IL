from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openpi.diffusion.policy import BrainCoDiffusionPolicy
    import openpi.policies.policy as _policy
    from openpi.training import config as _config
    import openpi.transforms as transforms

from openpi.diffusion.config import DiffusionTrainConfig


def create_trained_policy(
    train_config: _config.TrainConfig | DiffusionTrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy | BrainCoDiffusionPolicy:
    """Create the runtime specified by a self-describing BrainCo checkpoint."""
    if isinstance(train_config, DiffusionTrainConfig):
        from openpi.diffusion.checkpoint import load_policy

        if any(value is not None for value in (repack_transforms, sample_kwargs, default_prompt, norm_stats)):
            raise ValueError("DP preprocessing and sampling are defined by its checkpoint")
        return load_policy(checkpoint_dir)[1]

    import jax.numpy as jnp

    import openpi.models.model as _model
    import openpi.policies.policy as _policy
    import openpi.shared.download as download
    from openpi.training import checkpoints as _checkpoints
    import openpi.transforms as transforms

    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
