# Installation

## Supported platform

- Linux x86_64
- Python 3.10
- NVIDIA GPU with a driver compatible with the selected PyTorch CUDA build
- Conda or an existing `.venv`
- FFmpeg with H.264 encoding support

The tested environment uses PyTorch 2.8.0 and torchvision 0.23.0.

## Create the data environment

```bash
bash scripts/create_env.sh
conda activate ego2lerobot
python -m cli doctor --profile data --strict
```

If a `.venv` already exists at the repo root, activate it instead:

```bash
source .venv/bin/activate
python -m cli doctor --profile data --strict
```

NumPy, PyTorch, and the remaining packages use the USTC PyPI mirror by default.
Set `EGO2LEROBOT_TORCH_INDEX_URL` to use another compatible package index, or
`EGO2LEROBOT_TORCH_FIND_LINKS` for a trusted flat wheel mirror. `EGO2LEROBOT_PYPI_INDEX_URL`
overrides the general package source, and `EGO2LEROBOT_NUMPY_INDEX_URL` overrides only
NumPy.

If a download is interrupted, run the same command again. Only activate the
environment and run `python -m cli doctor` after the script prints that the environment
is ready.

## MANO assets

HaWoR and LeRobot export use the upstream `_DATA` layout:

```text
third_party/HaWoR/_DATA/data/mano/MANO_RIGHT.pkl
third_party/HaWoR/_DATA/data_left/mano_left/MANO_LEFT.pkl
```

Create an account on the [official MANO website](https://mano.is.tue.mpg.de/),
download `mano_v*_*.zip`, and accept the
[MANO license](https://mano.is.tue.mpg.de/license.html). MANO cannot be
downloaded or redistributed by this project.

## Data backends

GeoCalib, MoGe, and Mega-SAM source snapshots live under `third_party/`. An
authorized HaWoR checkout must be supplied locally at `third_party/HaWoR`; the
adapted infra copy is not part of source releases because HaWoR prohibits
redistribution of modifications.

After reviewing `THIRD_PARTY_NOTICES.md`:

```bash
bash scripts/install_data_backends.sh
python -m cli doctor --profile data --strict
```

The installer neither downloads missing source nor recreates internal
third-party adaptations. It also does not download model weights or MANO. Some
native CUDA extensions used by the processing chain must be compiled for the
installed PyTorch and GPU architecture.

Place separately obtained backend assets at:

```text
model/geocalib/pinhole.tar
model/moge2/model.pt
model/megasam/megasam_final.pth
model/hawor/hawor.ckpt
model/hawor/model_config.yaml
model/hawor/detector.pt
third_party/HaWoR/weights/external/droid.pth
third_party/HaWoR/thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth
```

These files remain Git-ignored. Their licenses are independent from the source
licenses summarized in `THIRD_PARTY_NOTICES.md`.
