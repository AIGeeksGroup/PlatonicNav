"""Shared helpers for asset-level pipeline variants."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from platonicnav.mapping.assets import load_episode_assets, load_segments
from platonicnav.mapping.background_cull import cull_background_nodes, remove_edges_touching_background
from platonicnav.mapping.graph_io import load_graph_json
from platonicnav.mapping.ptm import build_ptm_graph
from platonicnav.mapping.vision_encoder import build_node_embeddings
from platonicnav.schemas import EpisodeAssets, PTMBuildResult, PTMGraph, SegmentRecord, to_jsonable


@dataclass(frozen=True)
class PreparedAssets:
    config: dict[str, Any]
    config_path: Path
    assets: EpisodeAssets
    segments: list[SegmentRecord]
    graph: Any
    embedding_by_node: dict[int, Any]
    cull_result: Any
    ptm_result: PTMBuildResult
    planning_ptm: PTMGraph


def load_config(config_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping")
    data["_config_path"] = str(config_path.resolve())
    return data


def resolve_path(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n")


def prepare_mapping_stack(config: dict[str, Any] | Path) -> PreparedAssets:
    if isinstance(config, Path):
        config = load_config(config)
    config_path = Path(config.get("_config_path", "config.yaml")).resolve()
    base = config_path.parent
    manifest_path = resolve_path(base, config.get("episode_manifest"))
    if manifest_path is None:
        raise ValueError("config must define episode_manifest")
    assets = load_episode_assets(manifest_path)
    segments = load_segments(assets.segments_path)
    graph = load_graph_json(assets.graph_path)
    embedding_by_node = build_node_embeddings(
        segments,
        sidecar_path=assets.vision_embeddings_path,
        dim=int(config.get("vision_encoder", {}).get("dim", 64)),
    )

    cull_cfg = config.get("background_cull_config", {})
    background_terms = tuple(cull_cfg.get("background_terms", ["floor", "ceiling"]))
    cull_result = cull_background_nodes(
        segments,
        embedding_by_node,
        background_terms=background_terms,
        threshold=float(cull_cfg.get("threshold", 0.35)),
    )

    ptm_cfg = config.get("ptm", {})
    ptm_result = build_ptm_graph(
        graph,
        embedding_by_node,
        geo_weight_key=str(ptm_cfg.get("geo_weight_key", "d_geo_m")),
        edge_weight_key=str(config.get("edge_weight_key", "edge_weight_ptm")),
        lambda_g=float(ptm_cfg.get("lambda_g", 0.8)),
        lambda_s=float(ptm_cfg.get("lambda_s", 0.2)),
        quantile_q=float(ptm_cfg.get("quantile_q", 0.95)),
    )
    planning_graph = remove_edges_touching_background(
        ptm_result.ptm_graph.graph,
        cull_result.background_node_ids,
    )
    planning_ptm = PTMGraph(
        graph=planning_graph,
        graph_path=None,
        edge_weight_key=ptm_result.ptm_graph.edge_weight_key,
    )
    return PreparedAssets(
        config=config,
        config_path=config_path,
        assets=assets,
        segments=segments,
        graph=graph,
        embedding_by_node=embedding_by_node,
        cull_result=cull_result,
        ptm_result=ptm_result,
        planning_ptm=planning_ptm,
    )


def output_dir_for(prepared: PreparedAssets, *, default_subdir: str) -> Path:
    base = prepared.config_path.parent
    configured = resolve_path(base, prepared.config.get("output_dir"))
    output_dir = configured or (base / "outputs" / default_subdir / prepared.assets.episode_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def planner_edge_weight(prepared: PreparedAssets) -> str:
    planner_cfg = prepared.config.get("planner")
    if isinstance(planner_cfg, dict):
        return str(planner_cfg.get("edge_weight_key", prepared.planning_ptm.edge_weight_key))
    return prepared.planning_ptm.edge_weight_key

