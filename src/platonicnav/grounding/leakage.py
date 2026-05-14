"""No-leakage checks for blind-match pipelines."""

from __future__ import annotations

from typing import Any

FORBIDDEN_GOAL_KEYS = {
    "goal_position",
    "goal_positions",
    "goal_xyz",
    "goal_node",
    "goal_node_id",
    "goal_mask",
    "gt_mask",
    "gt_goal_mask",
    "goal_instance_id",
    "metadata_goal_position",
}


def assert_no_goal_metadata_leakage(record: Any, *, context: str = "record") -> None:
    if isinstance(record, dict):
        for key, value in record.items():
            if str(key) in FORBIDDEN_GOAL_KEYS:
                raise AssertionError(f"{context} contains forbidden goal metadata key: {key}")
            assert_no_goal_metadata_leakage(value, context=f"{context}.{key}")
    elif isinstance(record, list):
        for idx, value in enumerate(record):
            assert_no_goal_metadata_leakage(value, context=f"{context}[{idx}]")

