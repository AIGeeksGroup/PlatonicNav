# HM3D-IIN GTmask Pipeline

HM3D-IIN is the controlled PTM ablation.

The mapping, PTM edge construction, planner, and ObjectReact controller are
shared with the rest of PlatonicNav. The goal source is different: HM3D-IIN
uses GT mask + IoU goal grounding to match the native ObjectReact setting.

This variant does not use blind matching or language encoders.

Run the tiny asset-level example:

```bash
python scripts/run_iin_gtmask.py --config configs/iin_gtmask.yaml
```

Required additional asset:

- `goal_mask_path`: a JSON binary mask. Segment masks can be embedded in
  `segments.jsonl` or referenced by `mask_path`.
