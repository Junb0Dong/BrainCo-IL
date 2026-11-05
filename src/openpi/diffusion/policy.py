"""LeRobot 0.1 DP wrapper producing absolute action chunks for the existing inferencer."""

from __future__ import annotations

import time

from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType
from lerobot.configs.types import PolicyFeature
import numpy as np
import torch

from openpi.diffusion.config import CAMERAS
from openpi.diffusion.config import DiffusionTrainConfig
from openpi.diffusion.data import absolute_actions
from openpi.diffusion.data import arm_mask
from openpi.diffusion.data import preprocess_image
from openpi.diffusion.data import tensor_stats


def model_config(cfg: DiffusionTrainConfig, *, device: str | None = None) -> DiffusionConfig:
    m = cfg.model
    return DiffusionConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (m.action_dim,)),
            **{key: PolicyFeature(FeatureType.VISUAL, (3, cfg.image_size, cfg.image_size)) for key in CAMERAS},
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (m.action_dim,))},
        n_obs_steps=1,
        horizon=m.horizon,
        n_action_steps=m.action_horizon,
        down_dims=tuple(m.down_dims),
        crop_shape=None,
        crop_is_random=False,
        use_group_norm=True,
        pretrained_backbone_weights=None,
        use_separate_rgb_encoder_per_camera=False,
        noise_scheduler_type=m.noise_scheduler_type,
        num_train_timesteps=m.num_train_timesteps,
        num_inference_steps=m.num_inference_steps,
        drop_n_last_frames=m.horizon - 1,
        do_mask_loss_for_padding=True,
        device=device or cfg.training.device,
    )


def create_model(cfg: DiffusionTrainConfig, stats: dict, *, device: str | None = None) -> DiffusionPolicy:
    target = device or cfg.training.device
    return DiffusionPolicy(model_config(cfg, device=target), tensor_stats(stats)).to(target)


@torch.inference_mode()
def predict_relative_chunk(model: DiffusionPolicy, batch: dict) -> torch.Tensor:
    normalized = model.normalize_inputs(batch)
    normalized["observation.images"] = torch.stack([normalized[k] for k in model.config.image_features], dim=-4)
    return model.unnormalize_outputs({"action": model.diffusion.generate_actions(normalized)})["action"]


class BrainCoDiffusionPolicy:
    supports_rtc = False

    def __init__(self, cfg: DiffusionTrainConfig, model: DiffusionPolicy):
        self.cfg = cfg
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.mask = torch.as_tensor(arm_mask(cfg), device=self.device)

    def reset(self):
        self.model.reset()

    @torch.inference_mode()
    def infer(self, observation: dict) -> dict:
        started = time.perf_counter()
        state = np.asarray(observation["observation/state"], dtype=np.float32)
        if state.shape != (self.cfg.model.action_dim,) or not np.isfinite(state).all():
            raise ValueError("DP expects a finite selected 56D state")
        state_tensor = torch.as_tensor(state.copy(), device=self.device)
        batch = {"observation.state": state_tensor[None, None]}
        for key, source in zip(
            CAMERAS, ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"), strict=True
        ):
            batch[key] = preprocess_image(np.asarray(observation[source]), self.cfg.image_size)[None, None].to(
                self.device
            )
        preprocess_ms = (time.perf_counter() - started) * 1000
        relative = predict_relative_chunk(self.model, batch)
        actions = absolute_actions(relative[0], state_tensor, self.mask).float().cpu().numpy()
        if (
            actions.shape != (self.cfg.model.action_horizon, self.cfg.model.action_dim)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("DP generated invalid actions")
        return {
            "actions": actions,
            "policy_timing": {
                "preprocess_ms": preprocess_ms,
                "policy_total_ms": (time.perf_counter() - started) * 1000,
            },
        }
