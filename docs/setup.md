# Setup

PlatonicNav keeps separate conda environments because Habitat, ObjectReact,
ETPNav, SAM2, and DINOv3 have incompatible Python / CUDA requirements.

Environment definitions are in `envs/`:

- `shared-sam2map.yml`
- `objectreact-nav.yml`
- `etpnav.yml`
- `platonicnavtm-bm.yml`

Set `PLATONICNAV_ROOT` to the repository root before running scripts.

