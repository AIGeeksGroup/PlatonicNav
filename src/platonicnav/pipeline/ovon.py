"""Asset-level OVON ideal PlatonicNav pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from platonicnav.control.objectreact_handoff import build_controller_handoff
from platonicnav.grounding.blind_match import BlindMatchGoalGrounder
from platonicnav.grounding.clusters import kmeans_visual_clusters
from platonicnav.grounding.leakage import assert_no_goal_metadata_leakage
from platonicnav.grounding.vocab import ensure_goal_in_vocabulary, load_vocabulary
from platonicnav.mapping.assets import load_episode_assets, load_segments
from platonicnav.mapping.background_cull import cull_background_nodes, remove_edges_touching_background
from platonicnav.mapping.graph_io import load_graph_json, write_graph_json
from platonicnav.mapping.ptm import build_ptm_graph
from platonicnav.mapping.vision_encoder import build_node_embeddings
from platonicnav.planning.candidate_dijkstra import candidate_dijkstra
from platonicnav.schemas import GoalQuery, PTMGraph, to_jsonable


def load_config(config_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{config_path} must contain a YAML mapping")
    data["_config_path"] = str(config_path.resolve())
    return data


def _resolve(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n")


def run_ovon_pipeline(config: dict[str, Any] | Path) -> dict[str, Path]:
    if isinstance(config, Path):
        config = load_config(config)
    config_path = Path(config.get("_config_path", "configs/ovon_bm.yaml")).resolve()
    base = config_path.parent

    manifest_path = _resolve(base, config.get("episode_manifest"))
    if manifest_path is None:
        raise ValueError("OVON config must define episode_manifest")
    assets = load_episode_assets(manifest_path)
    assert_no_goal_metadata_leakage(assets.raw, context="episode_manifest")

    segments = load_segments(assets.segments_path)
    for segment in segments:
        assert_no_goal_metadata_leakage(segment.raw, context=f"segment[{segment.node_id}]")

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
    if not cull_result.foreground_node_ids:
        raise ValueError("DINOv3-based background cull removed all visual nodes")

    vocab_path = _resolve(base, config.get("vocabulary_path"))
    vocabulary = load_vocabulary(vocab_path, inline=config.get("vocabulary"))
    vocabulary = ensure_goal_in_vocabulary(vocabulary, assets.goal_category)

    cluster_cfg = config.get("clusterer", {})
    n_clusters = int(cluster_cfg.get("n_clusters", min(len(vocabulary), len(cull_result.foreground_node_ids))))
    clusters = kmeans_visual_clusters(
        node_ids=list(cull_result.foreground_node_ids),
        embedding_by_node=embedding_by_node,
        segments=segments,
        n_clusters=n_clusters,
        max_iter=int(cluster_cfg.get("max_iter", 50)),
    )

    grounder = BlindMatchGoalGrounder(
        vocabulary=vocabulary,
        language_config=config.get("language_encoder", {}),
        vocabulary_path=vocab_path,
    )
    grounding, blind_match = grounder.ground(
        GoalQuery(dataset="ovon", category=assets.goal_category, raw_text=assets.goal_category),
        clusters,
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
    plan = candidate_dijkstra(
        planning_ptm,
        start_node=assets.start_node,
        candidate_goal_nodes=grounding.candidate_goal_nodes,
        edge_weight=str(config.get("planner", {}).get("edge_weight_key", planning_ptm.edge_weight_key))
        if isinstance(config.get("planner"), dict)
        else planning_ptm.edge_weight_key,
    )
    handoff = build_controller_handoff(assets, plan)

    output_dir = _resolve(base, config.get("output_dir")) or (config_path.parent / "outputs" / assets.episode_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "grounding": output_dir / "grounding.json",
        "ptm_graph": output_dir / "ptm_graph.json",
        "plan": output_dir / "plan.json",
        "controller_handoff": output_dir / "controller_handoff.json",
        "run_summary": output_dir / "run_summary.json",
    }
    _write_json(paths["grounding"], {"goal_grounding": grounding, "blind_match": blind_match, "clusters": clusters})
    write_graph_json(planning_graph, paths["ptm_graph"])
    _write_json(paths["plan"], plan)
    _write_json(paths["controller_handoff"], handoff)
    _write_json(
        paths["run_summary"],
        {
            "episode": assets,
            "background_cull": cull_result,
            "ptm": ptm_result.diagnostics,
            "grounding": grounding,
            "plan": plan,
            "controller_handoff": handoff,
            "outputs": {k: str(v) for k, v in paths.items()},
        },
    )
    return paths

