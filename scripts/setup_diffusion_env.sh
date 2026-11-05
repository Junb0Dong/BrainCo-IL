#!/usr/bin/env bash
# Provision a separate cu128 environment while preserving the existing JAX env.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT=.venv-diffusion
export UV_HTTP_TIMEOUT=300
export UV_CONCURRENT_DOWNLOADS=2
uv sync --frozen
uv pip install --python .venv-diffusion/bin/python \
  -r requirements/diffusion-cu128-deps.txt --index-url https://pypi.org/simple
uv pip install --python .venv-diffusion/bin/python \
  --no-deps torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip freeze --python .venv-diffusion/bin/python > .venv-diffusion/requirements-resolved.txt
.venv-diffusion/bin/python -c 'import torch; assert "sm_120" in torch.cuda.get_arch_list(); print(torch.__version__)'
