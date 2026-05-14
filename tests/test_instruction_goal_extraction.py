"""Tests for R2R-CE instruction goal extraction."""

from pathlib import Path
import json

from platonicnav.grounding.instruction_goal_extractor import RuleBasedGoalExtractor
from platonicnav.pipeline.r2rce import run_r2rce_pipeline


def test_instruction_goal_extractor_gets_final_landmark() -> None:
    extractor = RuleBasedGoalExtractor()
    result = extractor.extract("Walk down the hallway and stop by the chair.")

    assert result.extracted_goal == "chair"
    assert result.method == "rule_based_final_goal_phrase"


def test_r2rce_tiny_pipeline_writes_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "_config_path": str(repo_root / "configs" / "test_r2rce.yaml"),
        "episode_manifest": str(repo_root / "examples" / "r2rce_tiny" / "episode_manifest.json"),
        "vocabulary_path": str(repo_root / "examples" / "r2rce_tiny" / "vocabulary.json"),
        "output_dir": str(tmp_path),
        "vision_encoder": {"provider": "deterministic", "dim": 3},
        "language_encoder": {"provider": "deterministic", "dim": 3},
        "goal_extractor_config": {"fallback_last_words": 4},
        "background_cull_config": {
            "background_terms": ["floor", "ceiling"],
            "threshold": 0.95,
        },
        "clusterer": {"n_clusters": 2, "max_iter": 10},
        "ptm": {
            "geo_weight_key": "d_geo_m",
            "lambda_g": 0.8,
            "lambda_s": 0.2,
            "quantile_q": 0.95,
        },
        "planner": {"name": "candidate_dijkstra", "edge_weight_key": "edge_weight_ptm"},
        "edge_weight_key": "edge_weight_ptm",
    }

    outputs = run_r2rce_pipeline(config)

    extraction = json.loads(outputs["instruction_goal"].read_text())
    plan = json.loads(outputs["plan"].read_text())
    assert extraction["extracted_goal"] == "chair"
    assert plan["selected_goal_node"] == 2
