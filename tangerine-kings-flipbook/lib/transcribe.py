"""WhisperX wrapper with file-hash content cache.

The actual whisperx import is lazy so unit tests can mock _run_whisperx
without needing the whisperx package or a GPU.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def transcribe(audio_path: Path, cache_dir: Path) -> dict:
    """Run WhisperX on an audio file with file-hash content cache.

    Returns:
        {"words": [{"w": str, "start": float, "end": float}, ...], "text": str}
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(audio_path.read_bytes()).hexdigest()[:16]
    cache_path = cache_dir / f"whisperx_{h}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    result = _run_whisperx(audio_path)
    cache_path.write_text(json.dumps(result))
    return result


def _run_whisperx(audio_path: Path) -> dict:
    """Invoke WhisperX. Lazy-imported so unit tests can mock it without GPU."""
    import whisperx

    device = "cuda"
    compute_type = "float16"
    model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    audio = whisperx.load_audio(str(audio_path))
    rough = model.transcribe(audio, batch_size=16)

    align_model, metadata = whisperx.load_align_model(
        language_code=rough["language"], device=device
    )
    aligned = whisperx.align(
        rough["segments"], align_model, metadata, audio, device,
        return_char_alignments=False,
    )

    words = []
    full_text_chunks = []
    for seg in aligned["segments"]:
        full_text_chunks.append(seg.get("text", "").strip())
        for word in seg.get("words", []):
            if "start" in word and "end" in word:
                words.append({
                    "w": word["word"].strip(),
                    "start": round(float(word["start"]), 3),
                    "end": round(float(word["end"]), 3),
                })
    return {"words": words, "text": " ".join(full_text_chunks).strip()}
