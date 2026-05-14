"""Tests for blind-match output schema."""

from platonicnav.grounding.blind_match import BlindMatchGoalGrounder
from platonicnav.schemas import GoalQuery, VisualCluster


def test_blind_match_returns_one_cluster_as_candidate_nodes() -> None:
    clusters = [
        VisualCluster("cluster_chair", (2,), (2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        VisualCluster("cluster_table", (3,), (0.0, 2.0, 0.0), (0.0, 1.0, 0.0)),
    ]
    grounder = BlindMatchGoalGrounder(
        vocabulary=["chair", "table"],
        language_config={"provider": "deterministic", "dim": 3},
    )
    grounder.embedding_overrides = {
        "chair": __import__("numpy").array([1.0, 0.0, 0.0], dtype="float32"),
        "table": __import__("numpy").array([0.0, 1.0, 0.0], dtype="float32"),
    }

    grounding, result = grounder.ground(GoalQuery(dataset="ovon", category="chair"), clusters)

    assert grounding.method == "category_level_blind_match_1_to_1"
    assert grounding.candidate_goal_nodes == (2,)
    assert result.matched_cluster_id == "cluster_chair"
    assert grounding.diagnostics["used_goal_metadata"] is False
