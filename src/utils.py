"""Shared utility functions."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent


def resolve_data_path(relative_path: str, config: dict | None = None) -> Path:
    """Resolve a data path relative to project root."""
    root = get_project_root()
    return root / relative_path
