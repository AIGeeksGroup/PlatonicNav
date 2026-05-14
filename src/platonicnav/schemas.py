"""Shared public schemas for the PlatonicNav pipelines.

The public repo exposes an asset-level pipeline: callers provide already
extracted episode assets, and PlatonicNav writes grounding, PTM, planning, and
controller-handoff records. These dataclasses are intentionally small so they
can be serialized cleanly and reused by OVON, HM3D-IIN, and R2R-CE frontends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoalQuery:
    dataset: str
    raw_text: str | None = None
    category: str | None = None
    instruction: str | None = None


@dataclass(frozen=True)
class GoalGrounding:
    method: str
    candidate_goal_nodes: tuple[int, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PTMGraph:
    graph: Any
    graph_path: Path | None = None
    edge_weight_key: str = "edge_weight_ptm"


@dataclass(frozen=True)
class NavigationPlan:
    selected_goal_node: int
    path: tuple[int, ...]
    path_cost: float


@dataclass(frozen=True)
class SegmentRecord:
    node_id: int
    frame_id: str | int | None = None
    bbox: tuple[float, ...] | None = None
    centroid: tuple[float, float, float] | None = None
    area: float | None = None
    semantic_hint: str | None = None
    embedding: tuple[float, ...] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeAssets:
    manifest_path: Path
    episode_id: str
    scene_id: str
    goal_category: str
    graph_path: Path
    segments_path: Path
    start_node: int
    instruction: str | None = None
    goal_mask_path: Path | None = None
    vision_embeddings_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualCluster:
    cluster_id: str
    node_ids: tuple[int, ...]
    centroid: tuple[float, float, float] | None
    embedding: tuple[float, ...]
    score: float = 0.0


@dataclass(frozen=True)
class BlindMatchResult:
    goal_text: str
    matched_cluster_id: str
    candidate_goal_nodes: tuple[int, ...]
    score: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstructionGoalExtraction:
    instruction: str
    extracted_goal: str
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GTMaskIoUResult:
    goal_node: int
    iou: float
    candidate_goal_nodes: tuple[int, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PTMBuildResult:
    ptm_graph: PTMGraph
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControllerHandoff:
    episode_id: str
    scene_id: str
    controller: str
    selected_goal_node: int
    path: tuple[int, ...]
    path_cost: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, Paths, tuples, and numpy scalars to JSON objects."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
