"""Tests for DINOv3-style PTM edge weights."""

import networkx as nx
import numpy as np

from platonicnav.mapping.ptm import build_ptm_graph


def test_ptm_edge_weights_are_non_negative_and_no_cross_modal_flags() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, d_geo_m=1.0)
    graph.add_edge(1, 2, d_geo_m=2.0)
    embeddings = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0], dtype=np.float32),
        2: np.array([1.0, 1.0], dtype=np.float32),
    }

    result = build_ptm_graph(graph, embeddings, lambda_g=0.8, lambda_s=0.2)

    for _, _, data in result.ptm_graph.graph.edges(data=True):
        assert data["edge_weight_ptm"] >= 0.0
        assert "d_pla" in data
        assert "d_pla_m" in data
    assert result.diagnostics["clip_used_for_ptm_edge_weight"] is False
    assert result.diagnostics["cross_modal_features_used_for_ptm_edge_weight"] is False

