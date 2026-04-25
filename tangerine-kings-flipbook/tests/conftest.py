from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def tiny_mp3(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "tiny.mp3"
    if not path.exists():
        pytest.skip(f"Fixture missing: {path}")
    return path
