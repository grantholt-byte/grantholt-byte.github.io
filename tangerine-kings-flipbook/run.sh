#!/usr/bin/env bash
# run.sh — single-command end-to-end render pipeline.
# Friend pastes one block, walks away, comes back to an MP4.
# Idempotent + resumable: re-running picks up after the last completed stage.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

mkdir -p output
SENTINEL_DIR="output/.stage_done"
mkdir -p "$SENTINEL_DIR"

stage() {
    local name="$1"; shift
    if [[ -f "$SENTINEL_DIR/$name" ]]; then
        echo "→ [skip] $name (already complete; delete $SENTINEL_DIR/$name to re-run)"
        return 0
    fi
    echo
    echo "═══ Stage: $name ═══"
    "$@"
    touch "$SENTINEL_DIR/$name"
}

# 0. Setup (no sentinel — setup.sh is itself idempotent)
if [[ ! -d .venv ]] || [[ ! -d vendor/ComfyUI ]]; then
    echo "═══ Stage: setup ═══"
    bash setup.sh
fi

# Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate

stage detect      python scripts/0_detect.py
stage analyze     python scripts/1_analyze_audio.py audio/TheFailing.mp3
stage schedule    python scripts/2_build_schedule.py
stage render      python scripts/3_render_keyframes.py
stage interpolate python scripts/4_interpolate_rife.py
stage assemble    python scripts/5_assemble_video.py

echo
echo "═══ Done ═══"
echo "Output: $REPO/output/the_failing_flipbook.mp4"
echo
echo "Send that file back to Grant however you want."
