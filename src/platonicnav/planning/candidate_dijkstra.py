"""Ideal-version candidate-goal Dijkstra planner."""

from __future__ import annotations

import math

import networkx as nx

from platonicnav.schemas import NavigationPlan, PTMGraph


def candidate_dijkstra(
    ptm_graph: PTMGraph,
    *,
    start_node: int,
    candidate_goal_nodes: tuple[int, ...],
    edge_weight: str | None = None,
) -> NavigationPlan:
    graph: nx.Graph = ptm_graph.graph
    weight_key = edge_weight or ptm_graph.edge_weight_key
    if int(start_node) not in graph:
        raise ValueError(f"start node {start_node} not present in graph")
    if not candidate_goal_nodes:
        raise ValueError("candidate_goal_nodes must be non-empty")

    best_goal: int | None = None
    best_path: list[int] | None = None
    best_cost = math.inf
    for goal in sorted({int(n) for n in candidate_goal_nodes}):
        if goal not in graph:
            continue
        try:
            cost, path = nx.single_source_dijkstra(graph, int(start_node), target=goal, weight=weight_key)
        except nx.NetworkXNoPath:
            continue
        if float(cost) < best_cost:
            best_goal = goal
            best_path = [int(n) for n in path]
            best_cost = float(cost)
    if best_goal is None or best_path is None:
        raise nx.NetworkXNoPath("no reachable candidate goal node")
    return NavigationPlan(
        selected_goal_node=int(best_goal),
        path=tuple(best_path),
        path_cost=float(best_cost),
    )

