"""Asset loading for the standalone public OVON pipeline."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from platonicnav.schemas import EpisodeAssets, SegmentRecord


def _read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def _resolve(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_episode_assets(manifest_path: Path) -> EpisodeAssets:
    manifest_path = manifest_path.resolve()
    data = _read_json(manifest_path)
    if not isinstance(data, dict):
        raise TypeError(f"{manifest_path} must contain a JSON object")
    base = manifest_path.parent
    goal_category = data.get("goal_category") or data.get("category") or data.get("goal_text")
    if not goal_category:
        goal_category = ""
    graph_path = _resolve(base, data.get("graph_path"))
    segments_path = _resolve(base, data.get("segments_path"))
    if graph_path is None or segments_path is None:
        raise ValueError("episode manifest must define graph_path and segments_path")
    return EpisodeAssets(
        manifest_path=manifest_path,
        episode_id=str(data.get("episode_id", manifest_path.stem)),
        scene_id=str(data.get("scene_id", "")),
        goal_category=str(goal_category),
        graph_path=graph_path,
        segments_path=segments_path,
        start_node=int(data["start_node"]),
        instruction=data.get("instruction"),
        goal_mask_path=_resolve(base, data.get("goal_mask_path") or data.get("gt_mask_path")),
        vision_embeddings_path=_resolve(base, data.get("vision_embeddings_path")),
        raw=data,
    )


def _tuple_or_none(value: Any, *, limit: int | None = None) -> tuple[float, ...] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).reshape(-1)
    if limit is not None:
        arr = arr[:limit]
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return tuple(float(v) for v in arr)


def segment_from_record(record: dict[str, Any]) -> SegmentRecord:
    node_id = record.get("node_id", record.get("id"))
    if node_id is None:
        raise ValueError(f"segment record missing node_id/id: {record}")
    centroid = (
        record.get("centroid")
        or record.get("world_centroid")
        or record.get("centroid_3d")
        or record.get("xyz")
    )
    return SegmentRecord(
        node_id=int(node_id),
        frame_id=record.get("frame_id"),
        bbox=_tuple_or_none(record.get("bbox")),
        centroid=_tuple_or_none(centroid, limit=3),  # type: ignore[arg-type]
        area=float(record["area"]) if record.get("area") is not None else None,
        semantic_hint=record.get("semantic_hint") or record.get("label"),
        embedding=_tuple_or_none(record.get("embedding")),
        raw=record,
    )


def load_segments(path: Path) -> list[SegmentRecord]:
    path = path.resolve()
    if path.suffix == ".jsonl":
        records: list[SegmentRecord] = []
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(segment_from_record(json.loads(line)))
        return records
    data = _read_json(path)
    raw_segments = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(raw_segments, list):
        raise TypeError(f"{path} must contain a segment list")
    return [segment_from_record(item) for item in raw_segments if isinstance(item, dict)]
