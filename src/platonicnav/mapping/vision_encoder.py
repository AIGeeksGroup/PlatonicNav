"""Vision embedding providers.

The public asset-level pipeline defaults to deterministic lightweight vectors
so tests can run without downloading DINOv3. Real DINOv3 features can be
provided in the segment manifest or an NPZ sidecar.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from platonicnav.schemas import SegmentRecord


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def deterministic_vector(key: str, *, dim: int = 64, namespace: str = "vision") -> np.ndarray:
    rng = np.random.default_rng(_stable_seed(f"{namespace}:{key}"))
    vec = rng.normal(size=dim).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def load_npz_embeddings(path: Path) -> dict[int, np.ndarray]:
    data = np.load(path)
    if "node_ids" not in data or "embeddings" not in data:
        raise ValueError(f"{path} must contain node_ids and embeddings arrays")
    node_ids = data["node_ids"].reshape(-1)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if embeddings.shape[0] != node_ids.shape[0]:
        raise ValueError("node_ids and embeddings length mismatch")
    return {int(node_ids[i]): embeddings[i].reshape(-1) for i in range(node_ids.shape[0])}


def build_node_embeddings(
    segments: list[SegmentRecord],
    *,
    sidecar_path: Path | None = None,
    dim: int = 64,
) -> dict[int, np.ndarray]:
    if sidecar_path is not None:
        return load_npz_embeddings(sidecar_path)
    embeddings: dict[int, np.ndarray] = {}
    for segment in segments:
        if segment.embedding is not None:
            embeddings[segment.node_id] = np.asarray(segment.embedding, dtype=np.float32)
        else:
            hint = segment.semantic_hint or "segment"
            embeddings[segment.node_id] = deterministic_vector(
                f"{hint}:{segment.node_id}:{segment.frame_id}", dim=dim
            )
    return embeddings

