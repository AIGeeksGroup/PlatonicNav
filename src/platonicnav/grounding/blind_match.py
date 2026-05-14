"""Category-level 1-to-1 blind matching for OVON."""

from __future__ import annotations

import numpy as np

from platonicnav.grounding.language_encoder import build_language_encoder
from platonicnav.grounding.vocab import load_vocabulary_embedding_overrides
from platonicnav.schemas import BlindMatchResult, GoalGrounding, GoalQuery, VisualCluster


def _normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def _distance_signature(x: np.ndarray) -> np.ndarray:
    z = _normalize(x.astype(np.float32))
    d = 1.0 - np.clip(z @ z.T, -1.0, 1.0)
    return np.sort(d, axis=1)


class BlindMatchGoalGrounder:
    """Map one object category name to one visual-segment cluster."""

    def __init__(
        self,
        *,
        vocabulary: list[str],
        language_config: dict | None = None,
        vocabulary_path=None,
    ) -> None:
        self.vocabulary = list(vocabulary)
        self.language_encoder = build_language_encoder(language_config)
        self.embedding_overrides = load_vocabulary_embedding_overrides(vocabulary_path)

    def _language_embeddings(self) -> np.ndarray:
        encoded = self.language_encoder.encode(self.vocabulary)
        if self.embedding_overrides:
            override_dim = int(next(iter(self.embedding_overrides.values())).reshape(-1).shape[0])
            if encoded.shape[1] != override_dim:
                # Keep deterministic fallback available for names without
                # explicit vectors while matching the asset-provided dimension.
                fallback = build_language_encoder({"provider": "deterministic", "dim": override_dim})
                encoded = fallback.encode(self.vocabulary)
        for idx, name in enumerate(self.vocabulary):
            override = self.embedding_overrides.get(name)
            if override is not None:
                encoded[idx] = override.reshape(-1)
        return encoded.astype(np.float32)

    def ground(self, query: GoalQuery, clusters: list[VisualCluster]) -> tuple[GoalGrounding, BlindMatchResult]:
        goal = query.category or query.raw_text
        if goal is None:
            raise ValueError("blind match requires query.category or query.raw_text")
        if goal not in self.vocabulary:
            raise ValueError(f"goal category {goal!r} missing from vocabulary")
        if not clusters:
            raise ValueError("blind match requires at least one visual cluster")
        cluster_embeddings = np.asarray([cluster.embedding for cluster in clusters], dtype=np.float32)
        language_embeddings = self._language_embeddings()
        c_sig = _distance_signature(cluster_embeddings)
        l_sig = _distance_signature(language_embeddings)
        goal_idx = self.vocabulary.index(goal)
        goal_sig = l_sig[goal_idx]
        width = min(c_sig.shape[1], goal_sig.shape[0])
        relational = np.linalg.norm(c_sig[:, :width] - goal_sig[:width][None, :], axis=1)

        # A tiny direct-space tie breaker is useful for deterministic assets
        # where the relational signature is underdetermined. It must not be
        # used as CLIP/VLM goal scoring; both sides are provided through the
        # blind-match encoder interfaces.
        c_norm = _normalize(cluster_embeddings)
        l_norm = _normalize(language_embeddings)
        direct = 1.0 - np.clip(c_norm @ l_norm[goal_idx], -1.0, 1.0)
        scores = relational + 1e-3 * direct
        best_idx = int(np.argmin(scores))
        matched = clusters[best_idx]
        grounding = GoalGrounding(
            method="category_level_blind_match_1_to_1",
            candidate_goal_nodes=tuple(int(n) for n in matched.node_ids),
            diagnostics={
                "goal_text": goal,
                "matched_cluster_id": matched.cluster_id,
                "score": float(scores[best_idx]),
                "relational_score": float(relational[best_idx]),
                "direct_tiebreak_score": float(direct[best_idx]),
                "n_clusters": len(clusters),
                "n_language_anchors": len(self.vocabulary),
                "used_goal_metadata": False,
                "used_gt_mask": False,
            },
        )
        return grounding, BlindMatchResult(
            goal_text=goal,
            matched_cluster_id=matched.cluster_id,
            candidate_goal_nodes=grounding.candidate_goal_nodes,
            score=float(scores[best_idx]),
            diagnostics=dict(grounding.diagnostics),
        )
