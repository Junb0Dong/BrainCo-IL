# BrainCo-IL

本仓库保留 BrainCo 使用的 PI0/PI0.5、ACT、Diffusion Policy、LeRobot 2.1 数据流水线、训练与
独立 Policy Inferencer 推理进程。

## 训练

```bash
cp configs/experiments/pi05.example.yaml my_experiment.yaml
# 将 base: pi05 改成 base: act 即切换 ACT

uv run python scripts/compute_norm_stats.py --config-path my_experiment.yaml
uv run python scripts/train.py my_experiment.yaml
```

简洁 experiment YAML 只写 dataset、语义 groups、action horizon 和训练参数；
joint name 与数值切片统一从 LeRobot `meta/info.json` 自动解析。

每个 step 自动生成 `train_config.yaml` 与 `policy_contract.json`。二者都标记
`generated: true`、`do_not_edit: true` 并带 hash，禁止手改。前者只供
BrainCo-IL 重建训练时的数据/模型流水线，后者供跨仓握手。

## 部署

生产入口只有独立 Python 推理进程：

```bash
uv run python -m openpi.deploy.policy_inferencer --checkpoint /path/to/checkpoint
```

Policy Inferencer 实现 DESCRIBE、LOAD/READY、INFER/RESULT、RESET/CLOSE。它接收带
joint name 的 observation，在仓库内部完成选维、图像预处理、归一化、模型
推理和反归一化，跨仓只返回按 group 打包的 absolute action chunk。

PI0.5 与 ACT 共用完全相同的部署协议，`revo_deploy` 不解析模型类型，也不
包含任何 JAX/PyTorch/LeRobot runtime。

## Diffusion Policy

DP 使用当前固定的 LeRobot 0.1 PyTorch 模型，入口仍为 `scripts/train.py`。
5090 训练使用独立 cu128 环境，不改变已有 JAX 环境：

```bash
bash scripts/setup_diffusion_env.sh
UV_PROJECT_ENVIRONMENT=.venv-diffusion CUDA_VISIBLE_DEVICES=4 \
  uv run --no-sync python scripts/train.py configs/experiments/diffusion.example.yaml
```

示例配置使用 pouring_set_junbo_1103 的原始 v2.1 数据，state/action 各选取 56 个具名关节。
三路 RGB 共用 256×256 resize；单帧观测，`model.horizon=32`，
`policy.action_horizon=16` 是实际返回的动作长度。手臂目标相对当前 state 编码，手部保持绝对值；
inferencer 反归一化并还原手臂绝对目标后返回四组动作。

episodes 0–44 训练、45–49 验证。训练自动计算 train-only MIN_MAX 统计，每个 episode 的末尾
31 个起点不采样。DP 不使用 ACT/π0.5 的 norm stats，也不消费任务文本或声明 RTC。

每个完整 step 的 `params/` 包含原生 LeRobot config/safetensors 和归一化参数；
`assets/diffusion_stats.json` 保存可审计的统计与划分。生成的 YAML、contract 和
`inference_manifest.json` 将接口及资产绑定到同一 checkpoint。`params/` 默认保存 EMA；
`train_state.pt` 另存 live/EMA、优化器、scheduler 和 RNG，部署时不传输。

```bash
# --steps 是最终累计 optimizer step；恢复后继续相同的采样顺序。
UV_PROJECT_ENVIRONMENT=.venv-diffusion CUDA_VISIBLE_DEVICES=4 \
  uv run --no-sync python scripts/train.py configs/experiments/diffusion.example.yaml \
  --resume checkpoints/diffusion/diffusion_pouring_set_1103_delta_obs1/last

UV_PROJECT_ENVIRONMENT=.venv-diffusion CUDA_VISIBLE_DEVICES=4 \
  uv run --no-sync python scripts/eval_diffusion.py \
  --checkpoint checkpoints/diffusion/diffusion_pouring_set_1103_delta_obs1/best \
  --output-json outputs/diffusion/evaluation.json
```

run 目录包含 `metrics.jsonl`、`status.json`、`best`/`last`。评估报告分别给出四组关节的绝对
误差、录制观测下相邻 chunk 的跳变，以及冷/热推理延迟；这些指标不代表真机任务成功率。
standard 调度在 chunk 用完后再请求推理，去噪延迟仍会产生等待。

环境安装、配置、后台任务与部署说明见 [Diffusion 接入说明](docs/diffusion_integration.md)。
