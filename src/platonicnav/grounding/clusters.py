"""Visual-segment clustering for category-level blind matching."""

from __future__ import annotations

import numpy as np

from platonicnav.schemas import SegmentRecord, VisualCluster


def _normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def _mean_centroid(segments: list[SegmentRecord], node_ids: list[int]) -> tuple[float, float, float] | None:
    by_node = {segment.node_id: segment for segment in segments}
    rows = [
        by_node[node_id].centroid
        for node_id in node_ids
        if node_id in by_node and by_node[node_id].centroid is not None
    ]
    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    mean = arr.mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def kmeans_visual_clusters(
    *,
    node_ids: list[int],
    embedding_by_node: dict[int, np.ndarray],
    segments: list[SegmentRecord],
    n_clusters: int,
    max_iter: int = 50,
) -> list[VisualCluster]:
    if not node_ids:
        return []
    rows = np.stack([np.asarray(embedding_by_node[int(n)], dtype=np.float32).reshape(-1) for n in node_ids])
    rows = _normalize(rows)
    k = max(1, min(int(n_clusters), len(node_ids)))
    init_idx = np.linspace(0, len(node_ids) - 1, num=k, dtype=int)
    centers = rows[init_idx].copy()
    labels = np.zeros(len(node_ids), dtype=int)
    for _ in range(max_iter):
        distances = ((rows[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for idx in range(k):
            members = rows[labels == idx]
            if members.size:
                centers[idx] = members.mean(axis=0)
        centers = _normalize(centers)
    clusters: list[VisualCluster] = []
    for idx in range(k):
        member_ids = [int(node_ids[i]) for i in np.where(labels == idx)[0]]
        if not member_ids:
            continue
        emb = rows[np.where(labels == idx)[0]].mean(axis=0)
        norm = float(np.linalg.norm(emb))
        if norm:
            emb = emb / norm
        clusters.append(
            VisualCluster(
                cluster_id=f"cluster_{idx}",
                node_ids=tuple(sorted(member_ids)),
                centroid=_mean_centroid(segments, member_ids),
                embedding=tuple(float(v) for v in emb.reshape(-1)),
            )
        )
    return clusters

