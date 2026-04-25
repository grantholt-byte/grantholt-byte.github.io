# Tangerine Kings — *The Failing* Flipbook

Self-contained pipeline that renders an animated music video for the Tangerine Kings track *The Failing* using Flux.1 [dev] + LoRA keyframes interpolated with RIFE 4.x. Runs entirely on local GPUs. No paid services, no Replicate API at runtime.

**Friend-facing setup:** open <https://grantholt-byte.github.io/tangerine-kings-flipbook/>.

## One-command run

```bash
git clone https://github.com/grantholt-byte/grantholt-byte.github.io.git
cd grantholt-byte.github.io/tangerine-kings-flipbook
export HF_TOKEN=hf_...
bash run.sh
```

`run.sh` is idempotent and resumable — re-running skips already-completed stages.

## Pipeline

```
0_detect.py           → output/hardware.json
1_analyze_audio.py    → output/timing_map.json          (librosa + WhisperX)
2_build_schedule.py   → output/keyframe_schedule.jsonl
3_render_keyframes.py → output/keyframes/*.png          (Flux + LoRA, multi-GPU)
4_interpolate_rife.py → output/frames/*.png             (RIFE 3×, 8→24 fps)
5_assemble_video.py   → output/the_failing_flipbook.mp4
```

## Project layout

| Path | What |
|---|---|
| `audio/TheFailing.mp3` | Source song (~4.6 MB, committed) |
| `config/shot_vocabulary.yaml` | 11 shots × 3 intensities = 33 prompt templates |
| `config/analyze_audio.yaml` | librosa thresholds + intensity bins |
| `config/schedule_rules.yaml` | Section rotations, hold timing, scream fallbacks |
| `lib/` | Pure-Python helpers (audio, transcribe, schedule, hardware, comfy, rife, post) |
| `scripts/` | 0_detect through 5_assemble — six sequential pipeline stages |
| `workflows/flux_lora_keyframe.json` | ComfyUI graph for Flux + LoRA inference |
| `tests/` | pytest suite (mocks WhisperX/ComfyUI/RIFE) |
| `setup.sh` | One-time install (Python venv, ComfyUI, Flux, LoRA, RIFE) |
| `run.sh` | End-to-end pipeline runner with stage-resume sentinels |
| `index.html` | GitHub Pages landing for the friend |
| `models/` | (gitignored) populated by `setup.sh` |
| `vendor/` | (gitignored) ComfyUI + RIFE clones |
| `output/` | (gitignored) all generated artifacts |

## Hardware auto-tuning

`scripts/0_detect.py` runs `nvidia-smi`, picks ComfyUI VRAM mode + Flux precision + batch size, and writes `output/hardware.json`. Downstream scripts read it.

| Smallest card VRAM | ComfyUI mode | Flux precision | Batch |
|---|---|---|---|
| ≥ 40 GB | `--highvram` | fp16 | 4 |
| ≥ 22 GB | `--highvram` | fp16 | 2 |
| ≥ 16 GB | `--normalvram` | fp8 | 1 |
| ≥ 12 GB | `--normalvram` | fp8 | 1 |
| < 12 GB | unsupported | — | — |

Multi-GPU rigs run one ComfyUI per card, dispatched round-robin.

## Iteration

- **Edit `config/shot_vocabulary.yaml`** to tune prompts. Delete `output/.stage_done/render` and re-run — only changed frames re-render (cache key = hash of prompt + seed + lora_strength).
- **Edit `output/timing_map.json`** by hand to fix section boundaries or scream timing if the auto-detection misfires. Delete `output/.stage_done/schedule` and `output/.stage_done/render`, re-run.
- **Edit `config/schedule_rules.yaml`** to change shot rotations per section. Same delete-and-re-run pattern.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Unit tests run without GPU. The librosa-dependent `tests/test_audio.py` is auto-skipped when librosa is unavailable (e.g., on macOS without LLVM toolchain).

## Design + plan

- Spec: <https://github.com/grantholt-byte/trump-emo/blob/master/docs/superpowers/specs/2026-04-25-thefailing-flipbook-design.md>
- Plan: <https://github.com/grantholt-byte/trump-emo/blob/master/docs/superpowers/plans/2026-04-25-thefailing-flipbook.md>
