"""Asset-level HM3D-IIN PlatonicNav-GTmask pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from platonicnav.control.objectreact_handoff import build_controller_handoff
from platonicnav.grounding.gtmask_iou import GTMaskIoUGrounder
from platonicnav.mapping.graph_io import write_graph_json
from platonicnav.pipeline.common import (
    output_dir_for,
    planner_edge_weight,
    prepare_mapping_stack,
    write_json,
)
from platonicnav.planning.candidate_dijkstra import candidate_dijkstra
from platonicnav.schemas import GoalQuery


def run_iin_gtmask_pipeline(config: dict[str, Any] | Path) -> dict[str, Path]:
    prepared = prepare_mapping_stack(config)
    if prepared.assets.goal_mask_path is None:
        raise ValueError("IIN GTmask config/manifest must define goal_mask_path")

    grounder = GTMaskIoUGrounder(
        goal_mask_path=prepared.assets.goal_mask_path,
        segment_base_dir=prepared.assets.segments_path.parent,
    )
    grounding, gtmask = grounder.ground(
        GoalQuery(dataset="hm3d_iin", category=prepared.assets.goal_category or None),
        prepared.segments,
        allowed_node_ids=prepared.cull_result.foreground_node_ids,
    )
    plan = candidate_dijkstra(
        prepared.planning_ptm,
        start_node=prepared.assets.start_node,
        candidate_goal_nodes=grounding.candidate_goal_nodes,
        edge_weight=planner_edge_weight(prepared),
    )
    handoff = build_controller_handoff(prepared.assets, plan)

    output_dir = output_dir_for(prepared, default_subdir="iin_gtmask")
    paths = {
        "grounding": output_dir / "grounding.json",
        "ptm_graph": output_dir / "ptm_graph.json",
        "plan": output_dir / "plan.json",
        "controller_handoff": output_dir / "controller_handoff.json",
        "run_summary": output_dir / "run_summary.json",
    }
    write_json(paths["grounding"], {"goal_grounding": grounding, "gtmask_iou": gtmask})
    write_graph_json(prepared.planning_ptm.graph, paths["ptm_graph"])
    write_json(paths["plan"], plan)
    write_json(paths["controller_handoff"], handoff)
    write_json(
        paths["run_summary"],
        {
            "episode": prepared.assets,
            "background_cull": prepared.cull_result,
            "ptm": prepared.ptm_result.diagnostics,
            "grounding": grounding,
            "plan": plan,
            "controller_handoff": handoff,
            "outputs": {k: str(v) for k, v in paths.items()},
        },
    )
    return paths

