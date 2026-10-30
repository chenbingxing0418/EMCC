"""Repository-local paths used by the ISCXVPN2016 experiments."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MY_MODEL_DIR = PROJECT_ROOT
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets"
DEFAULT_DATA_DIR = (
    DEFAULT_DATA_ROOT
    / "ISCXVPN2016"
    / "processed"
    / "prefix_1"
)
