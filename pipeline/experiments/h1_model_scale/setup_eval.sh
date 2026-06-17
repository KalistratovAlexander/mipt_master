#!/bin/bash
set -euo pipefail

# Eval machine setup for H1 evaluation (vast.ai, H100 80GB).
# Installs all dependencies needed by evaluate_unified.py and stat_tests.py.
# Idempotent — safe to run multiple times.
#
# Usage (from /workspace, before running run_h1.sh):
#   bash pipeline/evaluation/setup_eval.sh

if [ -f /tmp/.vast_eval_setup_done ]; then
    echo ">>> Eval dependencies already installed, skipping."
    return 0 2>/dev/null || exit 0
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo ">>> Installing eval dependencies..."
apt-get update -qq && apt-get install -y -qq build-essential python3-dev > /dev/null

pip3 install --upgrade pip -q
pip3 install torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121 -q
pip3 install "transformers>=4.51.0,<6" accelerate -q
pip3 install datasets pandas pyarrow polars -q
pip3 install sentence-transformers -q
pip3 install scipy numpy -q

# Flash Attention 2 (--attn-impl flash_attention_2)
pip3 install flash-attn --no-build-isolation -q 2>/dev/null && \
    echo ">>> Flash Attention 2: installed" || \
    echo ">>> Flash Attention 2: not available, using SDPA"

echo ">>> Verifying..."
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -c "
import torch, transformers, sentence_transformers, scipy
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
print(f'Transformers {transformers.__version__}')
print(f'sentence-transformers {sentence_transformers.__version__}')
print(f'scipy {scipy.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
"

touch /tmp/.vast_eval_setup_done
echo ">>> Eval setup complete."
