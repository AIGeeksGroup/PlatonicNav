# OVON Ideal Pipeline

OVON uses the full ideal PlatonicNav asset-level pipeline:

1. Build a Platonic Topological Map from RGB observations.
2. Use DINOv3 node embeddings and DINOv3-based background culling.
3. Ground the OVON category text with category-level blind matching.
4. Produce candidate goal nodes from the matched visual-segment cluster.
5. Run candidate-goal Dijkstra on the culled PTM.
6. Execute with the frozen ObjectReact controller.

No metadata goal position or GT mask is used for OVON goal grounding.

## Asset-Level Inputs

The public repo expects prepared assets instead of running Habitat directly:

- `episode_manifest.json`: episode id, scene id, goal category, graph path,
  segment path, and start node.
- `segments.jsonl`: segment/node records with optional cached embeddings.
- `graph.json`: ObjectReact-compatible topological graph with existing
  geometric edge distances under `d_geo_m`.
- `vocabulary.json` or `.txt`: object-category vocabulary, optionally with
  precomputed language embeddings for lightweight examples.

Run:

```bash
python scripts/run_ovon_bm.py --config configs/ovon_bm.yaml
```

Outputs are:

- `grounding.json`
- `ptm_graph.json`
- `plan.json`
- `controller_handoff.json`
- `run_summary.json`
