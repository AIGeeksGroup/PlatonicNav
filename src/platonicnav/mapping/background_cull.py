"""DINOv3-style background culling for asset-level PlatonicNav."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from platonicnav.mapping.vision_encoder import deterministic_vector
from platonicnav.schemas import SegmentRecord


@dataclass(frozen=True)
class BackgroundCullResult:
    foreground_node_ids: tuple[int, ...]
    background_node_ids: tuple[int, ...]
    scores: dict[int, float]
    diagnostics: dict[str, float | int | str | list[str]]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom == 0.0:
        return 0.0
    return float(np.clip(np.dot(aa, bb) / denom, -1.0, 1.0))


def cull_background_nodes(
    segments: list[SegmentRecord],
    embedding_by_node: dict[int, np.ndarray],
    *,
    background_terms: tuple[str, ...] = ("floor", "ceiling"),
    threshold: float = 0.35,
) -> BackgroundCullResult:
    if not segments:
        return BackgroundCullResult((), (), {}, {"status": "empty_segments"})
    dim = int(next(iter(embedding_by_node.values())).shape[0]) if embedding_by_node else 64
    prototypes = [
        deterministic_vector(term, dim=dim, namespace="vision-background")
        for term in background_terms
    ]
    bg_terms = {term.lower() for term in background_terms}
    scores: dict[int, float] = {}
    background: set[int] = set()
    all_nodes: set[int] = set()
    for segment in segments:
        node_id = int(segment.node_id)
        all_nodes.add(node_id)
        hint = (segment.semantic_hint or "").lower()
        emb = embedding_by_node.get(node_id)
        score = max((_cosine(emb, proto) for proto in prototypes), default=0.0) if emb is not None else 0.0
        scores[node_id] = float(score)
        if hint in bg_terms or score >= float(threshold):
            background.add(node_id)
    foreground = tuple(sorted(all_nodes.difference(background)))
    return BackgroundCullResult(
        foreground_node_ids=foreground,
        background_node_ids=tuple(sorted(background)),
        scores=scores,
        diagnostics={
            "mode": "dinov3_based_background_cull",
            "threshold": float(threshold),
            "background_terms": list(background_terms),
            "n_foreground": len(foreground),
            "n_background": len(background),
        },
    )


def remove_edges_touching_background(
    graph: nx.Graph,
    background_node_ids: tuple[int, ...],
    *,
    in_place: bool = False,
) -> nx.Graph:
    g = graph if in_place else graph.copy()
    bg = {int(n) for n in background_node_ids}
    edges_to_remove = [
        (u, v)
        for u, v in g.edges()
        if int(u) in bg or int(v) in bg
    ]
    g.remove_edges_from(edges_to_remove)
    g.graph["platonicnav_background_filter"] = {
        "mode": "remove_edges_touching_dinov3_background_nodes",
        "n_background_nodes": len(bg),
        "n_removed_edges": len(edges_to_remove),
    }
    return g

