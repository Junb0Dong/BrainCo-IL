#!/usr/bin/env bash
# Durable training/evaluation job for tmux. Failures retain logs and checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT=.venv-diffusion
export CUDA_VISIBLE_DEVICES="${DIFFUSION_GPU:-4}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
mkdir -p outputs/diffusion
run=checkpoints/diffusion/diffusion_pouring_set_1103_delta_obs1
resume_args=()
if [[ -f "$run/last/train_state.pt" ]]; then
  resume_args=(--resume "$run/last")
elif [[ -f checkpoints/diffusion/diffusion_pouring_1103_smoke200/200/train_state.pt ]]; then
  resume_args=(--resume checkpoints/diffusion/diffusion_pouring_1103_smoke200/200)
fi
exec > >(tee -a outputs/diffusion/background.log) 2>&1
date -Is
printf 'Starting/resuming DP training on GPU %s\n' "$CUDA_VISIBLE_DEVICES"
uv run --no-sync python scripts/train.py configs/experiments/diffusion.example.yaml "${resume_args[@]}" \
  2>&1 | tee -a outputs/diffusion/train16000.log
uv run --no-sync python scripts/eval_diffusion.py --checkpoint "$run/best" \
  --max-samples 256 --output-json outputs/diffusion/final-eval.json \
  2>&1 | tee outputs/diffusion/final-eval.log
uv run --no-sync python scripts/deploy_diffusion_checkpoint.py --checkpoint "$run/best" \
  --report-json outputs/diffusion/final-deployment.json \
  2>&1 | tee outputs/diffusion/final-deployment.log
printf '\nTraining, offline evaluation and revi inference validation complete. Reports: outputs/diffusion/\n'
date -Is
