"""Tests for candidate-goal Dijkstra planning."""

import networkx as nx

from platonicnav.planning.candidate_dijkstra import candidate_dijkstra
from platonicnav.schemas import PTMGraph


def test_candidate_dijkstra_selects_lowest_cost_candidate() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, edge_weight_ptm=5.0)
    graph.add_edge(0, 2, edge_weight_ptm=1.0)
    graph.add_edge(2, 3, edge_weight_ptm=1.0)

    plan = candidate_dijkstra(
        PTMGraph(graph=graph, edge_weight_key="edge_weight_ptm"),
        start_node=0,
        candidate_goal_nodes=(1, 3),
    )

    assert plan.selected_goal_node == 3
    assert plan.path == (0, 2, 3)
    assert plan.path_cost == 2.0
