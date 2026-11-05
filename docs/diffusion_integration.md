# BrainCo-IL Diffusion Policy 接入

2026-11-05 更新（UTC）。DP 复用固定 LeRobot commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，
模型使用 PyTorch 2.7.1+cu128 / torchvision 0.22.1+cu128 / diffusers 0.37.1。
基础依赖由 uv.lock 固定，cu128 覆盖版本由 setup 脚本和 requirements 固定，
实际完整包列表保存在 `.venv-diffusion/requirements-resolved.txt`。

## 安装与训练

从仓库根目录执行，需已安装 uv；无需使用仓库旁的 LeRobot checkout，依赖从固定 commit 安装：

```bash
bash scripts/setup_diffusion_env.sh
```

先将 `configs/experiments/diffusion.example.yaml` 的 `experiment.dataset` 改为本机数据集路径，
并按数据集调整 `data.train_episodes` 和 `data.eval_episodes`。默认划分为 0–44 / 45–49，
两组必须非空且不重叠。数据需为 LeRobot v2.1，含三路相机和四组共 56 个具名关节。

```bash
UV_PROJECT_ENVIRONMENT=.venv-diffusion CUDA_VISIBLE_DEVICES=4 \
  uv run --no-sync python scripts/train.py configs/experiments/diffusion.example.yaml

# --steps 为累计 optimizer step；--resume 指向完整数字目录或 last。
UV_PROJECT_ENVIRONMENT=.venv-diffusion CUDA_VISIBLE_DEVICES=4 \
  uv run --no-sync python scripts/train.py configs/experiments/diffusion.example.yaml \
  --resume checkpoints/diffusion/diffusion_pouring_set_1103_delta_obs1/last

UV_PROJECT_ENVIRONMENT=.venv-diffusion CUDA_VISIBLE_DEVICES=4 \
  uv run --no-sync python scripts/eval_diffusion.py \
  --checkpoint checkpoints/diffusion/diffusion_pouring_set_1103_delta_obs1/best \
  --output-json outputs/diffusion/evaluation.json
```

`model.horizon=32` 为训练预测窗口；`policy.action_horizon=16` 为实际返回的动作数。
训练自动计算仅含训练集的 MIN_MAX 统计，不使用 ACT/π0.5 的 norm stats。
每个 episode 的末尾 31 个起点不采样，避免跨 episode 或动作 padding。
`params/` 保存 EMA 推理权重；`train_state.pt` 保存 live/EMA、优化器、scheduler、RNG 以恢复训练。
`metrics.jsonl`、`status.json` 与 `best`/`last` 记录进度和模型选择。

## 已完成实验（2026-11-05）

正式训练已完成 16,000 次 optimizer update。按 2,048 个固定验证窗口上的 EMA loss 选择
第 11,000 步（0.0205165）；最后一步 EMA loss 为 0.0224006。最佳模型已通过实际注册的
`brainco_il` 入口完成 static / DESCRIBE / LOAD / INFER，并发布至：

```text
/home/revi/BrainCo/pi05_ckpt/junbo/diffusion_pouring_set_1103_delta_obs1/11000
```

推理包共 1,064,970,414 字节（按 manifest 中的配置、权重和统计计），源/目标 SHA-256 一致。
本地评估使用验证集的 256 个观测窗口，热推理 P50 100.9 ms、P95 102.6 ms，
返回 16 步、30 Hz 的绝对目标，对应 533.3 ms 动作序列。

| 关节组 | 绝对目标 MAE（数据集原始关节单位） |
| --- | ---: |
| 左臂 | 0.01551 |
| 右臂 | 0.05256 |
| 左手 | 0.01241 |
| 右手 | 0.06651 |

这些是录制观测上的离线指标；本次没有运行真机闭环。
升级后通过实际 registry 再次比较 ACT baseline、ACT dropout、π0.5 标准及 guided RTC，
新旧输出的所有关节组最大差值均为 0。

## 实现边界

- `openpi.diffusion` 负责配置、v2.1 dataset adapter、共享图像处理、统计、训练和推理制品。
- `scripts/train.py` 负责延迟分发；原有 JAX 实现移至 `openpi.training.train_jax`，原函数仍可导入。
- 公共 inferencer 复用 named observation adapter、协议和 action grouping，LOAD 才选择运行时。
- 当前四组共 56D。手臂整个 horizon 相对同一个当前 state；推理后加回原始 state，手部不加偏移。
- DP 归一化与 JAX transforms 分开。DP 当前仅支持 standard 调度，不消费任务文本或声明 RTC。

## 后台任务

以下脚本针对示例 pouring 实验，路径、目标步数与 tmux 会话名均已固定；其他实验请直接使用训练 CLI。
历史会话为 `brainco-diffusion`，保留 `train`、`log`、`finalize` 三个窗口。
训练与发布均已完成，窗口用于查看保留的日志：

```bash
tmux attach -t brainco-diffusion
# Ctrl-b d：退出查看，不停止任务。
tail -f /home/junbo/BrainCo-IL/outputs/diffusion/train16000.log
```

正式任务从已验收的 smoke step 200 继续到累计 16,000 次 optimizer update。
前 200 步仍处于相同的 500-step warmup，模型、数据划分、归一化与采样顺序一致。
正式验证扩大至 2,048 个固定窗口，因此重置 smoke 验证分数，不将两个验证样本集的分数混排。

后台程序负责评估 best EMA，上传完整推理包并执行 revi 离线 conformance。
`finalize` 是独立 watcher，训练窗口关闭后补齐尚未完成的评估/发布，避免依赖交互 SSH 会话。
脚本遇错返回失败并保留日志，不将失败标成完成；重试 finalizer 不会重训。
首次发布因实际 backend 尚未升级而停止；升级后复用了完整 staging，通过校验后发布。
`finalize.log` 保留首次错误，末尾的 `Already verified and published` 表示重试确认完成。

报告目录为 `outputs/diffusion/`：

- `smoke-eval.json` / `smoke-deployment.json`：200-step 流水线验收。
- `full-regression-fixed.log`、`final-dp-tests.log`：仓库回归与 DP 专项测试。
- `actual-backend-legacy-parity.json`：切换后实际 registry 的旧/新输出对比。
- `backend-switch-20261105.json` / `.before.yaml`：启动命令变更及原 registry 快照。
- `final-eval.json`：正式训练后的分组误差与延迟。
- `final-deployment.json`：源/目标 SHA-256、发布位置和部署 conformance。
- `final-path-conformance.json`：发布后的最终路径再次通过实际 backend 验收。
- `finalize.log`：后台收尾状态。正式完成须同时检查训练 `status.json` 和最终部署报告的 `ok`。

如果训练失败，检查日志后使用 `scripts/run_diffusion_background.sh` 从最后完整 checkpoint 恢复；
该脚本默认使用 GPU 4，可通过 `DIFFUSION_GPU` 调整。不会终止其他 GPU 作业。

## revi 升级与回退

版本目录：`/home/revi/BrainCo/BrainCo-IL-diffusion-20261104`。环境复制自当前部署环境后独立安装
新源码，原版本、原环境及原模型均保留。wrapper 清理 ROS shell 环境变量并可从 `/tmp` 启动。

切换前，使用独立 staging registry 验收 DP、ACT baseline、ACT dropout、π0.5。
旧/新 backend 对同一录制观测进行数值比较；候选 backend 接收反序 names。
π0.5 同时验证 guided RTC，确认能力和输出保持兼容。

仅将真实 registry 的 `backends.brainco_il.command` 切换为新 wrapper；其他 backend 不变。
原 registry 快照与切换报告放在新版本的 `validation/`。回退时将此 command 恢复为
快照中的原值，并重新开启需要回退的推理会话；不覆盖其他 backend 的同期修改。

如需回退，在 revi 上运行（report 文件名须未使用）：

```bash
release=/home/revi/BrainCo/BrainCo-IL-diffusion-20261104
"$release/.venv/bin/python" "$release/deployments/revi/switch_backend.py" \
  --rollback "$release/validation/backend-switch-20261105.json" \
  --report "$release/validation/backend-rollback.json"
```

模型统一发布到 `/home/revi/BrainCo/pi05_ckpt/junbo/<实验名>/<step>/`。
发布脚本只传 params、assets、生成配置和 manifest，在 staging 校验源/目标哈希及
static/describe/load/infer 后原子发布；已有不同内容的成品拒绝覆盖。
中断后可传 `--staging-checkpoint <远端.upload目录/step>` 续传并重新校验，避免重复上传大文件。
发布和验证均不启动 Actor、不发布控制指令。`smoke200` 制品仅用于接口验收。

`scripts/deploy_diffusion_checkpoint.py` 使用非交互 SSH 密钥/config 认证，也可通过
`--ssh-socket` 复用已认证的 SSH control socket，不依赖本机 Codex skill 文件。
执行前需在 revi 安装本仓库与独立环境、配置 `brainco_il` registry，并确认脚本顶部的
主机、release、数据集和发布路径符合目标环境。部署脚本不会自动安装或切换 backend。

## 验收覆盖

DP 集成版本在独立源码副本中通过仓库回归：73 passed、3 skipped、1 deselected（manual）。
独立 cu128 环境的 DP 专项测试为 13 passed；改动文件的 Ruff、shell 语法和依赖一致性检查通过。

DP 专项测试和现有仓库回归分别执行：

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
  .venv-diffusion/bin/python -m pytest -q src/openpi/diffusion/diffusion_test.py

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 \
  taskset -c 0-15 .venv/bin/python -m pytest -q -m 'not manual'
```

测试覆盖按名字选维、当前状态 delta 往返、训练/部署图像处理一致、train-only 统计、
不跨 episode 的窗口、forward/backward、EMA 保存重载、训练恢复的一致性、轻量 DESCRIBE、
真实 stdio 生命周期、partial read、错误名字/shape/NaN/hash，以及既有 JAX 策略回归。
数据 worker 使用 spawn，每个 worker 单独启用固定 LeRobot 与 datasets.Column 的兼容处理。

本机固定 JAX 0.5.3 / Orbax 0.11.13 的 CPU 恢复训练曾出现 native allocator abort，
独立进程也可复现。对照实验中关闭 donation 或复制恢复后的数组均通过；现在 CPU restore
完成后先创建新缓冲区，再执行保留 donation 的训练步骤。GPU restore 路径不变。
完整 train/resume 测试仍与其他测试在同一进程运行，并检查恢复前后的 checkpoint 确实生成。
诊断日志为 `jax-original-recheck.log`、`jax-no-donation.log`、`jax-restore-copy.log`。
旧、新 backend 对比须沿用原 JAX 编译缓存；本次三类旧模型及 π0.5 RTC 的分组输出差值均为 0。
