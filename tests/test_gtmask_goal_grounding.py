"""Tests for HM3D-IIN GTmask goal grounding."""

from pathlib import Path
import json

from platonicnav.grounding.gtmask_iou import GTMaskIoUGrounder
from platonicnav.mapping.assets import load_segments
from platonicnav.pipeline.iin import run_iin_gtmask_pipeline
from platonicnav.schemas import GoalQuery


def test_gtmask_grounding_uses_iou_without_language_or_blind_match() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example = repo_root / "examples" / "iin_tiny"
    segments = load_segments(example / "segments.jsonl")
    grounder = GTMaskIoUGrounder(goal_mask_path=example / "goal_mask.json")

    grounding, result = grounder.ground(GoalQuery(dataset="hm3d_iin"), segments)

    assert grounding.candidate_goal_nodes == (2,)
    assert result.goal_node == 2
    assert result.iou == 1.0
    assert grounding.diagnostics["used_gt_mask"] is True
    assert grounding.diagnostics["used_language_encoder"] is False
    assert grounding.diagnostics["used_blind_match"] is False


def test_iin_tiny_pipeline_writes_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "_config_path": str(repo_root / "configs" / "test_iin.yaml"),
        "episode_manifest": str(repo_root / "examples" / "iin_tiny" / "episode_manifest.json"),
        "output_dir": str(tmp_path),
        "vision_encoder": {"provider": "deterministic", "dim": 3},
        "background_cull_config": {
            "background_terms": ["floor", "ceiling"],
            "threshold": 0.95,
        },
        "ptm": {
            "geo_weight_key": "d_geo_m",
            "lambda_g": 0.8,
            "lambda_s": 0.2,
            "quantile_q": 0.95,
        },
        "planner": {"name": "candidate_dijkstra", "edge_weight_key": "edge_weight_ptm"},
        "edge_weight_key": "edge_weight_ptm",
    }

    outputs = run_iin_gtmask_pipeline(config)

    plan = json.loads(outputs["plan"].read_text())
    assert plan["selected_goal_node"] == 2
    grounding = json.loads(outputs["grounding"].read_text())
    assert grounding["goal_grounding"]["method"] == "gtmask_iou"
