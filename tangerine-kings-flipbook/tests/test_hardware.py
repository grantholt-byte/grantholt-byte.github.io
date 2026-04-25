from lib.hardware import recommend_settings


def test_recommend_settings_uses_highvram_for_big_cards():
    rec = recommend_settings([{"name": "RTX 4090", "vram_gb": 24}])
    assert rec["comfy_vram_mode"] == "--highvram"
    assert rec["flux_precision"] == "fp16"
    assert rec["parallelism"] == 1
    assert rec["batch_size"] >= 1


def test_recommend_settings_uses_normalvram_for_midsize():
    rec = recommend_settings([{"name": "RTX 3080", "vram_gb": 16}])
    assert rec["comfy_vram_mode"] == "--normalvram"
    assert rec["flux_precision"] == "fp8"


def test_recommend_settings_fails_below_threshold():
    rec = recommend_settings([{"name": "GTX 1080", "vram_gb": 8}])
    assert rec["unsupported"] is True
    assert "12 GB" in rec["reason"]


def test_recommend_settings_no_gpus_unsupported():
    rec = recommend_settings([])
    assert rec["unsupported"] is True
    assert "No NVIDIA GPU" in rec["reason"]


def test_recommend_settings_scales_parallelism_with_card_count():
    rec = recommend_settings([{"name": "RTX 4090", "vram_gb": 24}] * 3)
    assert rec["parallelism"] == 3


def test_recommend_settings_picks_lowest_common_tier():
    """Mixed rig: tier follows the smallest card."""
    rec = recommend_settings([
        {"name": "RTX 4090", "vram_gb": 24},
        {"name": "RTX 3080", "vram_gb": 12},
    ])
    assert rec["comfy_vram_mode"] == "--normalvram"
    assert rec["flux_precision"] == "fp8"


def test_recommend_settings_a100_tier():
    rec = recommend_settings([{"name": "A100", "vram_gb": 80}])
    assert rec["comfy_vram_mode"] == "--highvram"
    assert rec["flux_precision"] == "fp16"
    assert rec["batch_size"] == 4
