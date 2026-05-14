"""Tests preventing metadata or GT-mask leakage into BM pipelines."""

import pytest

from platonicnav.grounding.leakage import assert_no_goal_metadata_leakage


def test_no_goal_metadata_leakage_allows_category_text() -> None:
    assert_no_goal_metadata_leakage({"goal_category": "chair", "episode_id": "ep0"})


def test_no_goal_metadata_leakage_rejects_goal_position() -> None:
    with pytest.raises(AssertionError):
        assert_no_goal_metadata_leakage({"goal_position": [1.0, 2.0, 3.0]})
