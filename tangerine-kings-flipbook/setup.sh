#!/usr/bin/env bash
# setup.sh — one-time bootstrap for the Tangerine Kings flipbook pipeline.
# Idempotent: re-running skips already-completed steps.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

LORA_TAR_URL="https://replicate.delivery/xezq/1CxwvJUldDaSJpuXL9pFCTXrci1YEJr58e4RPR8DxEt20UPLA/trained_model.tar"
RIFE_REPO="https://github.com/hzwer/Practical-RIFE.git"
COMFYUI_REPO="https://github.com/comfyanonymous/ComfyUI.git"

echo "═══ Tangerine Kings flipbook setup ═══"
echo

# 1. Preflight (uses only stdlib — runs before deps install)
echo "[1/6] Running preflight checks..."
python3 scripts/0_detect.py || {
    echo
    echo "Preflight failed. Fix the issues above and re-run setup.sh."
    exit 1
}

# 2. Python venv + deps
echo
echo "[2/6] Creating Python venv and installing dependencies..."
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# 3. ComfyUI
echo
echo "[3/6] Setting up ComfyUI (portable)..."
if [[ ! -d vendor/ComfyUI ]]; then
    mkdir -p vendor
    git clone --depth 1 "$COMFYUI_REPO" vendor/ComfyUI
fi
pushd vendor/ComfyUI >/dev/null
pip install --quiet -r requirements.txt
popd >/dev/null

# 4. Flux.1 [dev]
echo
echo "[4/6] Downloading Flux.1 [dev] checkpoint (~24 GB, one-time)..."
mkdir -p vendor/ComfyUI/models/checkpoints vendor/ComfyUI/models/loras
if [[ ! -f vendor/ComfyUI/models/checkpoints/flux1-dev.safetensors ]]; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        cat <<EOF

ERROR: HF_TOKEN environment variable is not set.
  One-time:
    1. Visit https://huggingface.co/black-forest-labs/FLUX.1-dev and accept the license.
    2. Generate a token at https://huggingface.co/settings/tokens
    3. export HF_TOKEN=hf_...
    4. Re-run setup.sh
EOF
        exit 1
    fi
    python -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id='black-forest-labs/FLUX.1-dev',
    filename='flux1-dev.safetensors',
    local_dir='vendor/ComfyUI/models/checkpoints',
    local_dir_use_symlinks=False,
    token=os.environ['HF_TOKEN'],
)
"
fi

# 5. LoRA from Replicate CDN (no auth)
echo
echo "[5/6] Downloading + extracting Tangerine Kings LoRA..."
if [[ ! -f vendor/ComfyUI/models/loras/tangerine_kings.safetensors ]]; then
    curl -L --fail -o /tmp/tangerine_kings.tar "$LORA_TAR_URL"
    mkdir -p /tmp/tk_lora_extract
    tar -xf /tmp/tangerine_kings.tar -C /tmp/tk_lora_extract
    if [[ -f /tmp/tk_lora_extract/lora.safetensors ]]; then
        cp /tmp/tk_lora_extract/lora.safetensors vendor/ComfyUI/models/loras/tangerine_kings.safetensors
    elif [[ -f /tmp/tk_lora_extract/output/flux_train_replicate/lora.safetensors ]]; then
        cp /tmp/tk_lora_extract/output/flux_train_replicate/lora.safetensors vendor/ComfyUI/models/loras/tangerine_kings.safetensors
    else
        echo "ERROR: lora.safetensors not found in extracted .tar. Inspect /tmp/tk_lora_extract/"
        find /tmp/tk_lora_extract -name "*.safetensors"
        exit 1
    fi
    rm -rf /tmp/tangerine_kings.tar /tmp/tk_lora_extract
fi

# 6. RIFE
echo
echo "[6/6] Setting up RIFE 4.x..."
if [[ ! -d vendor/Practical-RIFE ]]; then
    git clone --depth 1 "$RIFE_REPO" vendor/Practical-RIFE
fi
echo "       NOTE: RIFE 4.x model weights download manually if needed."
echo "       See https://github.com/hzwer/Practical-RIFE for the weights link."

echo
echo "═══ Setup complete ═══"
echo "Next: bash run.sh"
