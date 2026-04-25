"""End-to-end smoke covering script wiring.

The librosa-dependent path (scripts 1 + 5 with cv2) only runs when those
deps are installed. Pure-Python paths (scripts 0, 2 with mock data) always run.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent


def _import_script(name: str):
    """Import a numbered script as a module to verify its imports resolve."""
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_scripts_import_cleanly():
    """Every script's imports resolve (catches missing deps + typos)."""
    for name in (
        "0_detect",
        "2_build_schedule",
        "3_render_keyframes",
        "4_interpolate_rife",
    ):
        _import_script(name)


def test_workflow_json_is_valid():
    wf = json.loads((REPO / "workflows" / "flux_lora_keyframe.json").read_text())
    # Verify required nodes
    class_types = {n["class_type"] for n in wf.values()}
    assert "CheckpointLoaderSimple" in class_types
    assert "LoraLoader" in class_types
    assert "KSampler" in class_types
    assert "SaveImage" in class_types
    # Verify the prompt placeholder exists for substitution
    placeholders = [
        n for n in wf.values()
        if n["class_type"] == "CLIPTextEncode" and n["inputs"].get("text") == "PROMPT_PLACEHOLDER"
    ]
    assert len(placeholders) == 1


def test_shot_vocabulary_has_all_required_shots():
    import yaml
    vocab = yaml.safe_load((REPO / "config" / "shot_vocabulary.yaml").read_text())

    assert "style_anchor" in vocab
    assert "lora_strength" in vocab
    assert "shots" in vocab
    assert len(vocab["shots"]) == 11
    for shot_id, shot in vocab["shots"].items():
        assert "base" in shot
        assert "calm" in shot
        assert "medium" in shot
        assert "heavy" in shot


def test_schedule_rules_yaml_loads():
    import yaml
    cfg = yaml.safe_load((REPO / "config" / "schedule_rules.yaml").read_text())
    assert "section_rotations" in cfg
    assert "scream_fallbacks" in cfg
    # Every rotation entry must point to shots that exist in the vocabulary
    vocab = yaml.safe_load((REPO / "config" / "shot_vocabulary.yaml").read_text())
    valid_shots = set(vocab["shots"].keys())
    for section, rotation in cfg["section_rotations"].items():
        for shot in rotation:
            assert shot in valid_shots, f"{section} references unknown shot {shot}"
    for shot in cfg["scream_fallbacks"]:
        assert shot in valid_shots


def test_analyze_audio_yaml_loads_with_expected_keys():
    import yaml
    cfg = yaml.safe_load((REPO / "config" / "analyze_audio.yaml").read_text())
    assert cfg["fps_keyframes"] == 8
    assert "calm" in cfg["intensity_bins"]
    assert "medium" in cfg["intensity_bins"]
    assert "heavy" in cfg["intensity_bins"]
    assert "scream_detection" in cfg


def test_2_build_schedule_runs_with_synthetic_timing_map(tmp_path: Path):
    """Drop in a synthetic timing_map.json, run script 2, validate schedule.jsonl."""
    work = tmp_path / "work"
    work.mkdir()
    output = work / "output"
    output.mkdir()
    config = work / "config"
    config.mkdir()

    # Symlink the real lib + scripts + workflows so imports resolve
    for d in ("lib", "scripts", "workflows", "audio"):
        (work / d).symlink_to(REPO / d)
    # Copy configs (might be edited by the test)
    for f in (REPO / "config").iterdir():
        shutil.copy2(f, config / f.name)

    # Synthetic timing map: 16 sec @ 8 fps = 128 keyframes, modest intensity
    timing_map = {
        "song": "synthetic",
        "duration_sec": 16.0,
        "fps_keyframes": 8,
        "intensity": [
            {"t": i * 0.125, "rms": 0.10 + (i % 32) / 100, "onsets": 0, "flatness": 0.1}
            for i in range(128)
        ],
        "sections": [
            {"start": 0.0, "end": 4.0, "label": "intro"},
            {"start": 4.0, "end": 8.0, "label": "verse_1"},
            {"start": 8.0, "end": 12.0, "label": "chorus_1"},
            {"start": 12.0, "end": 16.0, "label": "outro"},
        ],
        "words": [],
        "screams": [],
    }
    (output / "timing_map.json").write_text(json.dumps(timing_map))

    # Invoke through the work-dir symlink so __file__.parent.parent → work
    r = subprocess.run(
        [sys.executable, str(work / "scripts" / "2_build_schedule.py")],
        cwd=work,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, f"script 2 failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"

    schedule_path = output / "keyframe_schedule.jsonl"
    assert schedule_path.exists()

    lines = schedule_path.read_text().splitlines()
    assert len(lines) == 128
    for line in lines:
        e = json.loads(line)
        required = {"index", "t", "section", "shot", "intensity_label", "intensity_rms", "seed"}
        assert required.issubset(e.keys())


def test_setup_sh_syntax():
    """Verify setup.sh parses (don't execute it — it would download 24 GB)."""
    r = subprocess.run(
        ["bash", "-n", str(REPO / "setup.sh")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"setup.sh syntax error:\n{r.stderr}"


def test_run_sh_syntax():
    r = subprocess.run(
        ["bash", "-n", str(REPO / "run.sh")],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"run.sh syntax error:\n{r.stderr}"


def test_index_html_parses():
    import html.parser

    class V(html.parser.HTMLParser):
        def error(self, message):  # type: ignore
            raise AssertionError(message)

    V().feed((REPO / "index.html").read_text())
