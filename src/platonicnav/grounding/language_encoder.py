"""Language embedding providers for category-level blind matching."""

from __future__ import annotations

import numpy as np

from platonicnav.mapping.vision_encoder import deterministic_vector


class DeterministicLanguageEncoder:
    """Small deterministic encoder for tests and lightweight examples."""

    def __init__(self, *, dim: int = 64, namespace: str = "language") -> None:
        self.dim = int(dim)
        self.namespace = namespace

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.stack(
            [deterministic_vector(text.lower(), dim=self.dim, namespace=self.namespace) for text in texts],
            axis=0,
        ).astype(np.float32)


class SentenceTransformerEncoder:
    """Optional runtime adapter; imported only when requested."""

    def __init__(self, model_name: str = "sentence-transformers/gtr-t5-base", device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
            dtype=np.float32,
        )


def build_language_encoder(config: dict | None = None):
    config = config or {}
    provider = config.get("provider", "deterministic")
    if provider == "deterministic":
        return DeterministicLanguageEncoder(
            dim=int(config.get("dim", 64)),
            namespace=str(config.get("namespace", "language")),
        )
    if provider == "sentence-transformers":
        return SentenceTransformerEncoder(
            model_name=str(config.get("model_name", "sentence-transformers/gtr-t5-base")),
            device=config.get("device"),
        )
    raise ValueError(f"unknown language encoder provider: {provider}")

