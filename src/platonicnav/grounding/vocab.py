"""Vocabulary loading for OVON category-level blind matching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _names_from_obj(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj] if obj.strip() else []
    if isinstance(obj, list):
        names: list[str] = []
        for item in obj:
            if isinstance(item, dict) and "name" in item:
                names.append(str(item["name"]))
            else:
                names.extend(_names_from_obj(item))
        return names
    if isinstance(obj, dict):
        for key in ("categories", "object_categories", "vocab", "names"):
            if key in obj:
                return _names_from_obj(obj[key])
        if "name" in obj:
            return [str(obj["name"])]
        return [str(k) for k in obj.keys()]
    return []


def load_vocabulary(path: Path | None = None, *, inline: list[str] | None = None) -> list[str]:
    if inline is not None:
        names = inline
    elif path is not None:
        if path.suffix == ".txt":
            names = [line.strip() for line in path.read_text().splitlines()]
        else:
            names = _names_from_obj(json.loads(path.read_text()))
    else:
        names = []
    cleaned = sorted({name.strip() for name in names if name and name.strip()})
    if not cleaned:
        raise ValueError("vocabulary is empty")
    return cleaned


def load_vocabulary_embedding_overrides(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or path.suffix == ".txt":
        return {}
    data = json.loads(path.read_text())
    items = data.get("categories", data) if isinstance(data, dict) else data
    overrides: dict[str, np.ndarray] = {}
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and "name" in item and "embedding" in item:
                overrides[str(item["name"])] = np.asarray(item["embedding"], dtype=np.float32)
    elif isinstance(items, dict):
        for name, value in items.items():
            if isinstance(value, dict) and "embedding" in value:
                overrides[str(name)] = np.asarray(value["embedding"], dtype=np.float32)
            elif isinstance(value, list):
                overrides[str(name)] = np.asarray(value, dtype=np.float32)
    return overrides


def ensure_goal_in_vocabulary(vocabulary: list[str], goal: str) -> list[str]:
    if goal in vocabulary:
        return vocabulary
    return sorted(set(vocabulary + [goal]))

