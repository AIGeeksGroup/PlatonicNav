"""JSON graph I/O for asset-level PlatonicNav runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def load_graph_json(path: Path) -> nx.Graph:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    graph = nx.Graph()
    for node in data.get("nodes", []):
        node_data = dict(node)
        node_id = _pop_first(node_data, ("id", "node_id"))
        graph.add_node(node_id, **node_data)
    for edge in data.get("edges", []):
        edge_data = dict(edge)
        source = _pop_first(edge_data, ("source", "u", "from"))
        target = _pop_first(edge_data, ("target", "v", "to"))
        graph.add_edge(source, target, **edge_data)
    graph.graph.update(data.get("graph", {}))
    graph.graph["source_path"] = str(path)
    return graph


def _pop_first(data: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        if key in data:
            return int(data.pop(key))
    raise KeyError(f"missing one of {keys}")


def graph_to_json_dict(graph: nx.Graph) -> dict[str, Any]:
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        item = {"id": int(node_id)}
        item.update(_jsonable(attrs))
        nodes.append(item)
    edges = []
    for source, target, attrs in graph.edges(data=True):
        item = {"source": int(source), "target": int(target)}
        item.update(_jsonable(attrs))
        edges.append(item)
    return {"graph": _jsonable(dict(graph.graph)), "nodes": nodes, "edges": edges}


def write_graph_json(graph: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph_to_json_dict(graph), indent=2, sort_keys=True) + "\n")
