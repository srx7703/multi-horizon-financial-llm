#!/bin/bash
# Install HF transformers + PEFT + PyTorch/XLA stack on TPU v6e VM.
# Run on the TPU VM, NOT locally.
#
# Assumptions: TPU VM image comes with Python 3.10+, pip, and JAX/keras libs
# from earlier runs. We add the PyTorch/XLA stack alongside without removing
# anything.

set -e

echo "=== TPU setup: HF transformers + PEFT + PyTorch/XLA ==="
date

# 1. Base PyTorch + torch_xla for TPU v6e (Trillium)
# torch_xla 2.5+ supports v6e via PJRT
pip install -q --upgrade pip
pip install -q torch~=2.5.0 torch_xla[tpu]~=2.5.0 \
    -f https://storage.googleapis.com/libtpu-releases/index.html

# 2. HuggingFace stack
pip install -q --upgrade \
    transformers==4.45.2 \
    peft==0.13.2 \
    accelerate==1.0.1 \
    datasets==3.0.1 \
    sentencepiece protobuf

# 3. Sanity check: torch_xla sees the 8 TPU chips
python3 -c "
import os
os.environ.setdefault('PJRT_DEVICE', 'TPU')
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
print('device:', xm.xla_device())
print('world size:', xr.world_size())
print('runtime devices:', xr.global_runtime_device_count())
"

echo ""
echo "=== setup done ==="
