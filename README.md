# PlatonicNav: Unveiling Semantic Correspondence in Navigation with Platonic Topological Maps

Official implementation of **PlatonicNav: Unveiling Semantic Correspondence in Navigation with Platonic Topological Maps**.

> Embodied visual navigation, where an agent perceives a complex environment and acts to reach a goal from raw sensory input, underpins a wide range of applications such as household service robotics, assistive robotics, and large-scale autonomous exploration. However, recent attempts to unify vision-and-language navigation (VLN) and object goal navigation (ObjNav) remain at the level of architectural fusion, mixed-task training, and large vision-language pretraining, without examining whether independently trained vision and language encoders may already share a common semantic structure. Moreover, even object-centric topological maps still ground language goals through explicit cross-modal supervision such as CLIP or large vision-language models, leaving open whether such grounding is possible from a purely vision-built map. To address these challenges, we extend the *Platonic Representation Hypothesis* to embodied navigation and recast vision-only ObjNav, cross-modal ObjNav, and VLN as three different interfaces to the same object-centric semantic manifold. We further introduce **PlatonicNav**, a training-free framework whose **Platonic Topological Map** fuses geometric and semantic node distances from a self-supervised visual encoder, and grounds language goals via *blind matching* without any paired vision-language data. Extensive experiments on simulation benchmarks including HM3D-IIN, OVON, and R2R-CE on MP3D, together with deployment on Unitree Go2, which demonstrate that **PlatonicNav** generalizes across tasks, modalities, and embodiments without explicit cross-modal training.
>
> Code: https://github.com/AIGeeksGroup/PlatonicNav  
> Website: https://aigeeksgroup.github.io/PlatonicNav

### [Paper](https://github.com/AIGeeksGroup/PlatonicNav) | [Website](https://aigeeksgroup.github.io/PlatonicNav) | [Code](https://github.com/AIGeeksGroup/PlatonicNav)

<p align="center">
  <img src="assets/platonicnav.png" width="95%" alt="PlatonicNav overview">
</p>

## Overview

PlatonicNav includes three asset-level reference pipelines:

- **OVON**: category-level blind matching.
- **HM3D-IIN**: GT mask + IoU goal grounding for PTM ablation.
- **R2R-CE**: natural-language instruction goal extraction followed by blind matching.

## Installation

```bash
git clone https://github.com/AIGeeksGroup/PlatonicNav.git
cd PlatonicNav

conda create -n platonicnav-public python=3.10
conda activate platonicnav-public

pip install -e .
```

For full simulator-level experiments, see the environment files under `envs/`.

## Quick Start

```bash
python scripts/run_ovon_bm.py --config configs/ovon_bm.yaml
python scripts/run_iin_gtmask.py --config configs/iin_gtmask.yaml
python scripts/run_r2rce_bm.py --config configs/r2rce_bm.yaml
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

## Repository Structure

```text
src/platonicnav/
├── mapping/      # asset loading, graph I/O, DINOv3-style cull, PTM weights
├── grounding/    # blind matching, GTmask IoU, instruction goal extraction
├── planning/     # candidate-goal Dijkstra
├── control/      # frozen ObjectReact-style controller handoff
└── pipeline/     # OVON, HM3D-IIN, and R2R-CE pipeline runners
```

## Citation

```bibtex
@article{platonicnav2026,
  title={PlatonicNav: Unveiling Semantic Correspondence in Navigation with Platonic Topological Maps},
  author={TODO},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is released under the license specified in `LICENSE`.

## Acknowledgments

PlatonicNav builds on ideas and tooling from ObjectReact, Habitat, SAM2,
DINOv3, ETPNav, and open-vocabulary embodied navigation benchmarks.

