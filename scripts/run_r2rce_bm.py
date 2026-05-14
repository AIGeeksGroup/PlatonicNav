#!/usr/bin/env python
"""Thin entry point for the R2R-CE instruction blind-match pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from platonicnav.pipeline.r2rce import run_r2rce_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the asset-level R2R-CE PlatonicNav pipeline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "r2rce_bm.yaml",
        help="Path to an R2R-CE pipeline YAML config.",
    )
    args = parser.parse_args()
    outputs = run_r2rce_pipeline(args.config.resolve())
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
