"""Platonic Topological Map edge weighting."""

from __future__ import annotations

import networkx as nx
import numpy as np

from platonicnav.schemas import PTMBuildResult, PTMGraph

_EPS = 1e-12


def cosine_l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom == 0.0:
        return 1.0
    return float(1.0 - np.clip(np.dot(aa, bb) / denom, -1.0, 1.0))


def _geo_weight(data: dict, geo_weight_key: str) -> float:
    for key in (geo_weight_key, "d_geo_m", "e3d_avg", "weight", "cost"):
        if key in data:
            return float(data[key])
    return 1.0


def _safe_quantile(values: list[float], q: float) -> tuple[float, bool]:
    if not values:
        return 1.0, True
    scale = float(np.quantile(np.asarray(values, dtype=np.float64), q))
    if not np.isfinite(scale) or scale <= _EPS:
        return 1.0, True
    return scale, False


def build_ptm_graph(
    graph: nx.Graph,
    embedding_by_node: dict[int, np.ndarray],
    *,
    geo_weight_key: str = "d_geo_m",
    edge_weight_key: str = "edge_weight_ptm",
    lambda_g: float = 0.8,
    lambda_s: float = 0.2,
    quantile_q: float = 0.95,
) -> PTMBuildResult:
    if lambda_g < 0 or lambda_s < 0:
        raise ValueError("lambda_g and lambda_s must be non-negative")
    g = graph.copy()
    raw_values: dict[tuple[int, int], tuple[float, float]] = {}
    geo_values: list[float] = []
    pla_values: list[float] = []
    missing = 0
    for u, v, data in g.edges(data=True):
        geo = max(0.0, _geo_weight(data, geo_weight_key))
        eu = embedding_by_node.get(int(u))
        ev = embedding_by_node.get(int(v))
        if eu is None or ev is None:
            missing += 1
            pla = 0.0
        else:
            pla = max(0.0, cosine_l2_distance(eu, ev))
            pla_values.append(pla)
        geo_values.append(geo)
        raw_values[(int(u), int(v))] = (geo, pla)
        data["d_geo_m"] = geo
        data["d_pla"] = pla
        data["edge_weight_geo_raw"] = geo
        data["edge_weight_pla_raw"] = pla

    scale_geo, scale_geo_fallback = _safe_quantile(geo_values, quantile_q)
    scale_pla, scale_pla_fallback = _safe_quantile(pla_values, quantile_q)
    updated = 0
    for u, v, data in g.edges(data=True):
        geo, pla = raw_values[(int(u), int(v))]
        pla_norm = pla / scale_pla if scale_pla > _EPS else pla
        pla_m = scale_geo * pla_norm
        weight = max(0.0, float(lambda_g) * geo + float(lambda_s) * pla_m)
        data["d_pla_m"] = float(pla_m)
        data["edge_weight_pla_meter"] = float(pla_m)
        data["d_hybrid"] = weight
        data[edge_weight_key] = weight
        updated += 1

    diagnostics = {
        "mode": "ptm_existing_edges_dinov3_semantic_weight",
        "edge_weight_key": edge_weight_key,
        "geo_weight_key": geo_weight_key,
        "lambda_g": float(lambda_g),
        "lambda_s": float(lambda_s),
        "quantile_q": float(quantile_q),
        "scale_geo": float(scale_geo),
        "scale_pla": float(scale_pla),
        "scale_geo_fallback": bool(scale_geo_fallback),
        "scale_pla_fallback": bool(scale_pla_fallback),
        "updated_edges": int(updated),
        "missing_embedding_edges": int(missing),
        "clip_used_for_ptm_edge_weight": False,
        "cross_modal_features_used_for_ptm_edge_weight": False,
    }
    g.graph["platonicnav_ptm"] = diagnostics
    return PTMBuildResult(PTMGraph(graph=g, edge_weight_key=edge_weight_key), diagnostics)

