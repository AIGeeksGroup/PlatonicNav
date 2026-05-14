"""Asset-level R2R-CE PlatonicNav pipeline with instruction goal extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from platonicnav.control.objectreact_handoff import build_controller_handoff
from platonicnav.grounding.blind_match import BlindMatchGoalGrounder
from platonicnav.grounding.clusters import kmeans_visual_clusters
from platonicnav.grounding.instruction_goal_extractor import RuleBasedGoalExtractor
from platonicnav.grounding.leakage import assert_no_goal_metadata_leakage
from platonicnav.grounding.vocab import ensure_goal_in_vocabulary, load_vocabulary
from platonicnav.mapping.graph_io import write_graph_json
from platonicnav.pipeline.common import (
    output_dir_for,
    planner_edge_weight,
    prepare_mapping_stack,
    resolve_path,
    write_json,
)
from platonicnav.planning.candidate_dijkstra import candidate_dijkstra
from platonicnav.schemas import GoalQuery


def run_r2rce_pipeline(config: dict[str, Any] | Path) -> dict[str, Path]:
    prepared = prepare_mapping_stack(config)
    assert_no_goal_metadata_leakage(prepared.assets.raw, context="episode_manifest")
    for segment in prepared.segments:
        assert_no_goal_metadata_leakage(segment.raw, context=f"segment[{segment.node_id}]")
    if not prepared.assets.instruction:
        raise ValueError("R2R-CE manifest must define instruction")

    extractor_cfg = prepared.config.get("goal_extractor_config", {})
    extractor = RuleBasedGoalExtractor(
        fallback_last_words=int(extractor_cfg.get("fallback_last_words", 4))
    )
    extraction = extractor.extract(prepared.assets.instruction)

    base = prepared.config_path.parent
    vocab_path = resolve_path(base, prepared.config.get("vocabulary_path"))
    vocabulary = load_vocabulary(vocab_path, inline=prepared.config.get("vocabulary"))
    vocabulary = ensure_goal_in_vocabulary(vocabulary, extraction.extracted_goal)

    cluster_cfg = prepared.config.get("clusterer", {})
    n_clusters = int(cluster_cfg.get("n_clusters", min(len(vocabulary), len(prepared.cull_result.foreground_node_ids))))
    clusters = kmeans_visual_clusters(
        node_ids=list(prepared.cull_result.foreground_node_ids),
        embedding_by_node=prepared.embedding_by_node,
        segments=prepared.segments,
        n_clusters=n_clusters,
        max_iter=int(cluster_cfg.get("max_iter", 50)),
    )
    grounder = BlindMatchGoalGrounder(
        vocabulary=vocabulary,
        language_config=prepared.config.get("language_encoder", {}),
        vocabulary_path=vocab_path,
    )
    grounding, blind_match = grounder.ground(
        GoalQuery(
            dataset="r2r_ce",
            raw_text=extraction.extracted_goal,
            category=extraction.extracted_goal,
            instruction=prepared.assets.instruction,
        ),
        clusters,
    )
    plan = candidate_dijkstra(
        prepared.planning_ptm,
        start_node=prepared.assets.start_node,
        candidate_goal_nodes=grounding.candidate_goal_nodes,
        edge_weight=planner_edge_weight(prepared),
    )
    handoff = build_controller_handoff(prepared.assets, plan)

    output_dir = output_dir_for(prepared, default_subdir="r2rce_bm")
    paths = {
        "instruction_goal": output_dir / "instruction_goal.json",
        "grounding": output_dir / "grounding.json",
        "ptm_graph": output_dir / "ptm_graph.json",
        "plan": output_dir / "plan.json",
        "controller_handoff": output_dir / "controller_handoff.json",
        "run_summary": output_dir / "run_summary.json",
    }
    write_json(paths["instruction_goal"], extraction)
    write_json(paths["grounding"], {"goal_grounding": grounding, "blind_match": blind_match, "clusters": clusters})
    write_graph_json(prepared.planning_ptm.graph, paths["ptm_graph"])
    write_json(paths["plan"], plan)
    write_json(paths["controller_handoff"], handoff)
    write_json(
        paths["run_summary"],
        {
            "episode": prepared.assets,
            "instruction_goal": extraction,
            "background_cull": prepared.cull_result,
            "ptm": prepared.ptm_result.diagnostics,
            "grounding": grounding,
            "plan": plan,
            "controller_handoff": handoff,
            "outputs": {k: str(v) for k, v in paths.items()},
        },
    )
    return paths

