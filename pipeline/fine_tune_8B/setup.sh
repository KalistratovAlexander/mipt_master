#!/bin/bash
set -euo pipefail

# Common setup for both stages on vast.ai (Qwen3-8B)
# Installs all dependencies once. Idempotent.
#
# Called automatically by stage1/run.sh and stage2/run.sh

if [ -f /tmp/.vast_setup_done ]; then
    echo ">>> Dependencies already installed, skipping."
    return 0 2>/dev/null || exit 0
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> Installing dependencies..."
apt-get update -qq && apt-get install -y -qq build-essential python3-dev > /dev/null

# libcuda symlink (sometimes missing in Docker)
[ ! -f /usr/lib/x86_64-linux-gnu/libcuda.so ] && \
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so 2>/dev/null || true

pip3 install --upgrade pip -q
pip3 install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121 -q
pip3 install "transformers>=4.51.0" -q
pip3 install trl datasets numpy bitsandbytes -q

# Unsloth (needed for Stage 2)
pip3 install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git" -q
pip3 uninstall torchao -y 2>/dev/null || true

# Flash Attention 2 (Ampere+ GPUs, ~2x speedup for 8B)
pip3 install flash-attn --no-build-isolation -q 2>/dev/null && \
    echo ">>> Flash Attention 2: installed" || \
    echo ">>> Flash Attention 2: not available, using SDPA"

# --- Verify ---
echo ">>> Verifying..."
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -c "
import torch, transformers
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
print(f'Transformers {transformers.__version__}')
from unsloth import FastLanguageModel; print('Unsloth OK')
"

touch /tmp/.vast_setup_done
echo ">>> Setup complete."
