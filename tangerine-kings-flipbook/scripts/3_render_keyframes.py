"""Render Flux+LoRA keyframes for the schedule, multi-GPU.

Reads output/hardware.json (written by scripts/0_detect.py) for parallelism,
ComfyUI vram mode, and batch size. Spawns N ComfyUI subprocess instances on
ports 8188..8188+N-1, dispatches keyframes round-robin. Caches outputs by
hash(prompt + seed + lora_strength).

Usage:
    python scripts/3_render_keyframes.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.comfy import ComfyClient, Dispatcher
from lib.hardware import load_hardware_json


def launch_comfy(
    repo: Path, port: int, gpu_index: int, vram_mode: str
) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    log_path = repo / f"output/comfy_gpu{gpu_index}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            sys.executable, str(repo / "vendor/ComfyUI/main.py"),
            "--port", str(port),
            "--listen", "127.0.0.1",
            "--disable-auto-launch",
            vram_mode,  # --highvram / --normalvram / --lowvram
        ],
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )


def wait_for_comfy(port: int, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(f"http://localhost:{port}/system_stats", timeout=2.0)
            return
        except httpx.HTTPError:
            time.sleep(1.0)
    raise TimeoutError(f"ComfyUI on port {port} did not come up within {timeout}s")


def build_prompt_text(entry: dict, vocabulary: dict) -> str:
    shot = vocabulary["shots"][entry["shot"]]
    base = shot["base"]
    intensity_text = shot[entry["intensity_label"]]
    return f"{vocabulary['style_anchor']} {base} {intensity_text}"


def cache_key(prompt_text: str, seed: int, lora_strength: float) -> str:
    h = hashlib.sha256()
    h.update(prompt_text.encode())
    h.update(str(seed).encode())
    h.update(str(lora_strength).encode())
    return h.hexdigest()[:16]


def main() -> None:
    repo = Path(__file__).parent.parent
    output_dir = repo / "output"
    keyframes_dir = output_dir / "keyframes"
    cache_dir = output_dir / "_keyframe_cache"
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    hardware = load_hardware_json(output_dir / "hardware.json")
    if hardware.get("unsupported"):
        sys.exit(f"Hardware unsupported: {hardware['reason']}")

    parallelism = hardware["parallelism"]
    vram_mode = hardware["comfy_vram_mode"]
    print(f"Rendering with {parallelism}-way parallelism, ComfyUI mode {vram_mode}")

    schedule_path = output_dir / "keyframe_schedule.jsonl"
    schedule = [
        json.loads(line) for line in schedule_path.read_text().splitlines() if line.strip()
    ]

    vocab = yaml.safe_load((repo / "config" / "shot_vocabulary.yaml").read_text())
    workflow = json.loads((repo / "workflows" / "flux_lora_keyframe.json").read_text())
    negative = vocab.get("negative", "blurry, photograph, photorealistic, 3d render")
    lora_strength = vocab.get("lora_strength", 0.85)

    print(f"Launching {parallelism} ComfyUI instance(s)...")
    procs: list[subprocess.Popen] = []
    clients: list[ComfyClient] = []
    try:
        for i in range(parallelism):
            port = 8188 + i
            procs.append(launch_comfy(repo, port, gpu_index=i, vram_mode=vram_mode))
            wait_for_comfy(port)
            clients.append(ComfyClient(host="localhost", port=port, timeout=600.0))
            print(f"  GPU {i} → port {port} ready")

        work = []
        cache_hits = 0
        for entry in schedule:
            prompt_text = build_prompt_text(entry, vocab)
            key = cache_key(prompt_text, entry["seed"], lora_strength)
            cache_path = cache_dir / f"{key}.png"
            out_path = keyframes_dir / f"{entry['index']:06d}.png"
            if cache_path.exists():
                shutil.copy2(cache_path, out_path)
                cache_hits += 1
                continue
            work.append({
                "prompt_text": prompt_text,
                "negative_text": negative,
                "seed": entry["seed"],
                "tag": f"{entry['index']:06d}|{key}",
            })

        print(f"Cache hits: {cache_hits} / {len(schedule)}; rendering {len(work)} new keyframes")

        if work:
            dispatcher = Dispatcher(clients)
            results = dispatcher.render(workflow, work)
            for tag, img_bytes in results.items():
                idx_str, key = tag.split("|")
                cache_path = cache_dir / f"{key}.png"
                cache_path.write_bytes(img_bytes)
                shutil.copy2(cache_path, keyframes_dir / f"{idx_str}.png")

        print(f"Done. {len(schedule)} keyframes in {keyframes_dir}")

    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
