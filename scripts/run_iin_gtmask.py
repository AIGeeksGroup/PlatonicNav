#!/usr/bin/env python
"""Thin entry point for the HM3D-IIN GTmask PTM pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from platonicnav.pipeline.iin import run_iin_gtmask_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the asset-level HM3D-IIN GTmask PlatonicNav pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "iin_gtmask.yaml",
        help="Path to an IIN GTmask pipeline YAML config.",
    )
    args = parser.parse_args()
    outputs = run_iin_gtmask_pipeline(args.config.resolve())
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
