# Third-party notices

This repository interfaces with upstream research software and model assets.
Those works are not relicensed under this project's MIT license.

| Component | Upstream | License notes | Distributed here |
| --- | --- | --- | --- |
| GeoCalib | `cvg/GeoCalib` | Apache-2.0; weights are CC BY 4.0 | Source only |
| MoGe | `microsoft/MoGe` | MIT plus bundled Apache-2.0 components; model terms may differ | Source only |
| Mega-SAM | `mega-sam/mega-sam` | Apache-2.0 for software; CC BY 4.0 for other materials | Source with disclosed local memory patches |
| HaWoR | `ThunderVVV/HaWoR` | CC BY-NC-ND 4.0; non-commercial and no redistributed modifications | No; modified infra copy is local and Git-ignored |
| MANO | Official MANO website | Separate registration and license agreement | No |
| PyTorch3D | `facebookresearch/pytorch3d` | BSD-3-Clause | No |

The three redistributable data-pipeline source trees are copied under
`third_party/` with their original license files. The current development
machine also has the infra HaWoR source at `third_party/HaWoR`, but that copy
contains a local path adaptation and is excluded from Git because the HaWoR
license prohibits redistribution of modified material. No model weights,
MANO files, generated native extensions, or training data are part of the
source snapshot. HaWoR commercial use or redistribution of this modified copy
requires separate permission from its authors. MANO always requires a separate
agreement with its owner.

Before publishing a release:

1. Record and review the imported third-party snapshot revision.
2. Retain upstream copyright and notice files.
3. Confirm that model checkpoint terms permit the intended use.
4. Keep MANO and other registration-gated files outside source distributions.
