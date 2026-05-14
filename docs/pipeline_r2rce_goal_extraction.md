# R2R-CE Goal Extraction Pipeline

R2R-CE uses the same ideal PlatonicNav-BM stack as OVON, with one extra
front-end stage:

1. Extract a final goal or landmark phrase from the natural-language
   instruction.
2. Treat that phrase as the language query for blind matching.
3. Ground it to visual candidate goal nodes.
4. Plan over the PTM and execute with the frozen ObjectReact controller.

This is a short-horizon semantic-goal bridge, not a full VLN solver.

Run the tiny asset-level example:

```bash
python scripts/run_r2rce_bm.py --config configs/r2rce_bm.yaml
```

The public implementation uses a transparent rule-based extractor for the
final goal / landmark phrase. Stronger parsers can replace it while preserving
the same `instruction_goal.json` output schema.
