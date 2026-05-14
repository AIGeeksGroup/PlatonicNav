"""End-to-end asset-level OVON pipeline test."""

from pathlib import Path
import json

from platonicnav.pipeline.ovon import run_ovon_pipeline


def test_ovon_tiny_pipeline_writes_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config = {
        "_config_path": str(repo_root / "configs" / "test_ovon.yaml"),
        "episode_manifest": str(repo_root / "examples" / "ovon_tiny" / "episode_manifest.json"),
        "vocabulary_path": str(repo_root / "examples" / "ovon_tiny" / "vocabulary.json"),
        "output_dir": str(tmp_path),
        "vision_encoder": {"provider": "deterministic", "dim": 3},
        "language_encoder": {"provider": "deterministic", "dim": 3},
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

    outputs = run_ovon_pipeline(config)

    assert set(outputs) == {
        "grounding",
        "ptm_graph",
        "plan",
        "controller_handoff",
        "run_summary",
    }
    for path in outputs.values():
        assert path.exists()
    plan = json.loads(outputs["plan"].read_text())
    assert plan["selected_goal_node"] == 2
    assert plan["path"][0] == 0
