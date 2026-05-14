"""GT mask + IoU goal grounding for HM3D-IIN asset-level runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from platonicnav.schemas import GTMaskIoUResult, GoalGrounding, GoalQuery, SegmentRecord


def _mask_from_obj(obj: Any) -> np.ndarray:
    if isinstance(obj, dict):
        if "mask" in obj:
            return _mask_from_obj(obj["mask"])
        shape = obj.get("shape") or obj.get("size")
        if "indices" in obj and shape is not None:
            mask = np.zeros(tuple(int(v) for v in shape), dtype=bool)
            for item in obj["indices"]:
                if isinstance(item, list) and len(item) == 2:
                    mask[int(item[0]), int(item[1])] = True
                else:
                    mask.reshape(-1)[int(item)] = True
            return mask
    arr = np.asarray(obj)
    if arr.ndim != 2:
        raise ValueError("mask must be a 2D array or an indices+shape object")
    return arr.astype(bool)


def load_mask(path: Path) -> np.ndarray:
    return _mask_from_obj(json.loads(path.read_text()))


def _segment_mask(segment: SegmentRecord, *, base_dir: Path | None = None) -> np.ndarray | None:
    raw = segment.raw
    if "mask" in raw:
        return _mask_from_obj(raw["mask"])
    path_value = raw.get("mask_path")
    if path_value:
        path = Path(path_value)
        if base_dir is not None and not path.is_absolute():
            path = base_dir / path
        return load_mask(path)
    return None


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


class GTMaskIoUGrounder:
    """Select the segment node with maximum IoU against the GT goal mask."""

    def __init__(self, *, goal_mask_path: Path, segment_base_dir: Path | None = None) -> None:
        self.goal_mask_path = goal_mask_path
        self.segment_base_dir = segment_base_dir
        self.goal_mask = load_mask(goal_mask_path)

    def ground(
        self,
        query: GoalQuery,
        segments: list[SegmentRecord],
        *,
        allowed_node_ids: tuple[int, ...] | None = None,
    ) -> tuple[GoalGrounding, GTMaskIoUResult]:
        allowed = {int(n) for n in allowed_node_ids} if allowed_node_ids is not None else None
        best_node: int | None = None
        best_iou = -1.0
        scored: list[dict[str, float | int]] = []
        for segment in segments:
            node_id = int(segment.node_id)
            if allowed is not None and node_id not in allowed:
                continue
            mask = _segment_mask(segment, base_dir=self.segment_base_dir)
            if mask is None:
                continue
            score = _iou(mask, self.goal_mask)
            scored.append({"node_id": node_id, "iou": score})
            if score > best_iou:
                best_node = node_id
                best_iou = score
        if best_node is None:
            raise ValueError("no segment masks available for GTmask IoU grounding")
        grounding = GoalGrounding(
            method="gtmask_iou",
            candidate_goal_nodes=(best_node,),
            diagnostics={
                "dataset": query.dataset,
                "goal_mask_path": str(self.goal_mask_path),
                "best_iou": float(best_iou),
                "used_gt_mask": True,
                "used_language_encoder": False,
                "used_blind_match": False,
                "scored_nodes": scored,
            },
        )
        return grounding, GTMaskIoUResult(
            goal_node=best_node,
            iou=float(best_iou),
            candidate_goal_nodes=(best_node,),
            diagnostics=dict(grounding.diagnostics),
        )

