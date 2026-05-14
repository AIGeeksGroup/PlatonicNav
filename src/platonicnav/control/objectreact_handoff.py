"""Asset-level handoff record for the frozen ObjectReact controller."""

from __future__ import annotations

from platonicnav.schemas import ControllerHandoff, EpisodeAssets, NavigationPlan


def build_controller_handoff(
    assets: EpisodeAssets,
    plan: NavigationPlan,
    *,
    controller: str = "frozen_objectreact_controller_handoff",
) -> ControllerHandoff:
    return ControllerHandoff(
        episode_id=assets.episode_id,
        scene_id=assets.scene_id,
        controller=controller,
        selected_goal_node=plan.selected_goal_node,
        path=plan.path,
        path_cost=plan.path_cost,
        diagnostics={
            "runtime_scope": "asset_level_closed_loop",
            "controller_frozen": True,
        },
    )

