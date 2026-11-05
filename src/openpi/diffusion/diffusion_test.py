import dataclasses
import io
import json
import os
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from openpi.deploy import policy_inferencer
from openpi.deploy import protocol
from openpi.diffusion import train as training
from openpi.diffusion.checkpoint import build_contract
from openpi.diffusion.checkpoint import load_policy
from openpi.diffusion.checkpoint import save_checkpoint
from openpi.diffusion.checkpoint import verify_artifact
from openpi.diffusion.config import CAMERAS
from openpi.diffusion.config import config_yaml
from openpi.diffusion.config import load_config
from openpi.diffusion.data import absolute_actions
from openpi.diffusion.data import arm_mask
from openpi.diffusion.data import compute_stats
from openpi.diffusion.data import preprocess_image
from openpi.diffusion.data import relative_actions
from openpi.diffusion.policy import BrainCoDiffusionPolicy
from openpi.diffusion.policy import create_model
from openpi.diffusion.train import WindowBatchSampler


@pytest.fixture
def cfg(tmp_path):
    torch.set_num_threads(2)
    root = tmp_path / "data"
    (root / "meta").mkdir(parents=True)
    names = [
        *(f"eef{i}" for i in range(14)),
        *(f"left_arm_joint{i}" for i in range(1, 8)),
        *(f"right_arm_joint{i}" for i in range(1, 8)),
        *(f"left_finger{i}_joint" for i in range(21)),
        *(f"right_finger{i}_joint" for i in range(21)),
    ]
    features = {key: {"shape": [70], "names": names} for key in ("observation.state", "action")}
    features.update({key: {"shape": [32, 40, 3], "dtype": "video"} for key in CAMERAS})
    info = {
        "codebase_version": "v2.1",
        "robot_type": "revotron_revo3",
        "fps": 30,
        "features": features,
        "chunks_size": 1000,
        "data_path": "episode_{episode_index}.parquet",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": e, "length": 12}) for e in (0, 1)))
    for episode in (0, 1):
        states = np.arange(12 * 70, dtype=np.float32).reshape(12, 70) / 100
        # Deliberately extreme held-out values reveal any stats leakage.
        states += 1000 * episode
        actions = states + 0.25
        pq.write_table(
            pa.table(
                {
                    "observation.state": states.tolist(),
                    "action": actions.tolist(),
                    "timestamp": (np.arange(12) / 30).tolist(),
                }
            ),
            root / f"episode_{episode}.parquet",
        )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "base": "diffusion",
                "experiment": {"name": "test", "dataset": str(root)},
                "policy": {"action_horizon": 4},
                "model": {"horizon": 8, "down_dims": [16, 32, 64], "num_inference_steps": 2},
                "data": {"image_size": 32, "train_episodes": [0], "eval_episodes": [1]},
                "training": {
                    "device": "cpu",
                    "precision": "fp32",
                    "batch_size": 2,
                    "microbatch_size": 1,
                    "num_workers": 0,
                    "num_steps": 2,
                    "checkpoint_base_dir": str(tmp_path / "ckpt"),
                },
            }
        )
    )
    return load_config(experiment)


def observation(cfg):
    return {
        "observation/state": np.arange(56, dtype=np.float32) / 10,
        **{
            key: np.zeros((40, 48, 3), np.uint8)
            for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image")
        },
    }


def test_configuration_and_contract(cfg, tmp_path):
    assert cfg.model.action_dim == 56
    assert cfg.interface["state_indices"] == tuple(range(14, 70))
    contract = build_contract(cfg, "test_step2")
    assert contract["chunk_steps"] == 4
    assert contract["inference_strategies"] == ["standard"]
    path = tmp_path / "runtime.yaml"
    path.write_text(config_yaml(cfg, "test_step2"))
    loaded = load_config(path)
    assert loaded.model.horizon == 8
    payload = yaml.safe_load(path.read_text())
    payload["train_config"]["image_size"] = 64
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(path)


def test_delta_uses_current_state_for_whole_chunk_and_preserves_hands(cfg):
    state = np.arange(56, dtype=np.float32)
    actions = np.stack([state + 1, state + 2, state + 3])
    relative = relative_actions(actions, state, arm_mask(cfg))
    np.testing.assert_allclose(relative[:, :14], np.broadcast_to(np.array([1, 2, 3])[:, None], (3, 14)))
    np.testing.assert_array_equal(relative[:, 14:], actions[:, 14:])
    np.testing.assert_array_equal(absolute_actions(relative, state, arm_mask(cfg)), actions)


def test_stats_transform_windows_and_exclude_held_out(cfg):
    stats = compute_stats(cfg)
    assert stats["state_count"] == 5
    assert stats["action_count"] == 40
    assert max(stats["stats"]["observation.state"]["max"]) < 10
    assert stats["stats"]["action"]["max"][0] == pytest.approx(5.15, abs=1e-5)
    assert stats["stats"]["action"]["min"][0] == pytest.approx(0.25, abs=1e-5)


def test_training_and_deployment_image_preprocess_match():
    image = np.random.default_rng(42).integers(0, 256, (37, 59, 3), dtype=np.uint8)
    dataset_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    torch.testing.assert_close(preprocess_image(image, 32), preprocess_image(dataset_image, 32), rtol=0, atol=0)
    with pytest.raises(ValueError, match="uint8"):
        preprocess_image(image.astype(np.float32), 32)


def test_sampler_resume_does_not_depend_on_prefetch():
    full = [i for batch in WindowBatchSampler(11, 42, 0, 40, 4) for i in batch]
    resumed = [i for batch in WindowBatchSampler(11, 42, 16, 40, 2) for i in batch]
    assert resumed == full[16:]
    assert len(set(full[:11])) == 11


def test_model_backward_chunk_and_save_load(cfg):
    stats = compute_stats(cfg)
    model = create_model(cfg, stats)
    batch = {
        "observation.state": torch.ones(2, 1, 56),
        "action": torch.ones(2, 8, 56),
        "action_is_pad": torch.zeros(2, 8, dtype=torch.bool),
        **{key: torch.rand(2, 1, 3, 32, 32) for key in CAMERAS},
    }
    loss, _ = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.diffusion.parameters() if p.grad is not None)
    wrapped = BrainCoDiffusionPolicy(cfg, model)
    torch.manual_seed(12)
    expected = wrapped.infer(observation(cfg))["actions"]
    path = save_checkpoint(cfg, model, stats, 2)
    _, restored = load_policy(path, device="cpu")
    torch.manual_seed(12)
    actual = restored.infer(observation(cfg))["actions"]
    assert actual.shape == (4, 56)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    restored.reset()
    assert all(not queue for queue in restored.model._queues.values())
    (path / "params/config.json").write_text("{}")
    with pytest.raises(ValueError, match="modified inference asset"):
        verify_artifact(path)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "nan"])
def test_named_io_rejects_invalid_inputs(cfg, failure):
    contract = build_contract(cfg, "test")
    groups = {
        g: {"names": list(names), "positions": np.zeros(len(names), np.float32)}
        for g, names in contract["required_joint_groups"].items()
    }
    if failure == "missing":
        del groups["left_arm"]
    elif failure == "duplicate":
        groups["left_arm"]["names"][0] = groups["left_arm"]["names"][1]
    else:
        groups["left_arm"]["positions"][0] = np.nan
    with pytest.raises(ValueError, match="joint group"):
        policy_inferencer._build_policy_observation({"joint_groups": groups}, contract)


def test_real_stdio_lifecycle_without_jax(cfg):
    model = create_model(cfg, compute_stats(cfg))
    checkpoint = save_checkpoint(cfg, model, compute_stats(cfg), 1)
    contract = build_contract(cfg, "test_step1")
    script = "import sys; from openpi.deploy.policy_inferencer import main; main(); assert 'jax' not in sys.modules"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with subprocess.Popen(
        [sys.executable, "-c", script, "--checkpoint", str(checkpoint)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd="/tmp",
    ) as process:

        def request(message):
            protocol.send_frame(process.stdin, message)
            return protocol.recv_frame(process.stdout)

        assert request({"type": "DESCRIBE"})["type"] == "CONTRACT"
        assert request({"type": "LOAD"})["type"] == "READY"
        groups = {
            g: {"names": names[::-1], "positions": np.arange(len(names), dtype=np.float32)[::-1]}
            for g, names in contract["required_joint_groups"].items()
        }
        obs = {"joint_groups": groups, "images": {k: np.zeros((40, 48, 3), np.uint8) for k in CAMERAS}}
        result = request({"type": "INFER", "request_id": 9, "observation": obs})
        assert result["type"] == "RESULT", result
        assert result["request_id"] == 9
        assert result["actions"]["left_hand"].shape == (4, 21)
        assert request({"type": "RESET"})["type"] == "RESET_DONE"
        assert request({"type": "CLOSE"})["type"] == "CLOSED"
        assert process.wait(timeout=20) == 0, process.stderr.read().decode()


def test_protocol_partial_reads_and_output_validation(cfg):
    class PartialStream(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 2))

    stream = PartialStream()
    protocol.send_frame(stream, {"type": "TEST", "value": np.ones((4, 56), np.float32)})
    stream.seek(0)
    np.testing.assert_array_equal(protocol.recv_frame(stream)["value"], np.ones((4, 56)))
    contract = build_contract(cfg, "test")
    for actions in (np.zeros((8, 56)), np.full((4, 56), np.nan), np.zeros((4, 55))):
        with pytest.raises(ValueError, match=r"policy|contract"):
            policy_inferencer._group_actions(actions, contract)


def test_describe_does_not_import_model_runtime(cfg, tmp_path):
    (tmp_path / "policy_contract.json").write_text(json.dumps(build_contract(cfg, "describe")))
    code = (
        "import sys; from openpi.deploy.policy_inferencer import main; main(); "
        "assert 'torch' not in sys.modules; assert 'jax' not in sys.modules"
    )
    stream = io.BytesIO()
    protocol.send_frame(stream, {"type": "DESCRIBE"})
    protocol.send_frame(stream, {"type": "CLOSE"})
    result = subprocess.run(
        [sys.executable, "-c", code, "--checkpoint", str(tmp_path)],
        input=stream.getvalue(),
        capture_output=True,
        timeout=15,
        cwd="/tmp",
        check=True,
    )
    assert protocol.recv_frame(io.BytesIO(result.stdout))["type"] == "CONTRACT"


def test_resume_matches_uninterrupted_training(cfg, monkeypatch):
    class NumericDataset:
        def __init__(self, config, *, train):
            self.config = config
            self.size = 5 if train else 3

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            return {
                "observation.state": torch.full((1, 56), index / 10),
                "action": torch.full((8, 56), index / 10),
                "action_is_pad": torch.zeros(8, dtype=torch.bool),
                **{key: torch.full((1, 3, 32, 32), index / 10) for key in CAMERAS},
            }

    monkeypatch.setattr(training, "DiffusionDataset", NumericDataset)
    cfg = dataclasses.replace(
        cfg,
        training=dataclasses.replace(
            cfg.training, save_interval=1, eval_interval=1, log_interval=1, max_eval_samples=2
        ),
    )
    training.train(cfg)
    resumed = dataclasses.replace(cfg, name="resumed")
    training.train(resumed, resume=str(cfg.checkpoint_dir / "1"))
    expected = torch.load(cfg.checkpoint_dir / "2/train_state.pt", weights_only=False)
    actual = torch.load(resumed.checkpoint_dir / "2/train_state.pt", weights_only=False)
    for kind in ("model", "ema"):
        for key in expected[kind]:
            torch.testing.assert_close(actual[kind][key], expected[kind][key], rtol=0, atol=0)
    assert actual["scheduler"] == expected["scheduler"]
