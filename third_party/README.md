# Bundled research backends

Ego2LeRobot keeps the redistributable upstream source snapshots used by its internal
data pipeline under this directory. This is not every adaptation required to
recreate the production pipeline. The current working-tree snapshot was
imported from `wuji-data-infra` at revision
`f198a1ed08b508f7bef7de70ae20b5fe29122997` on 2026-08-16. Upstream copyright,
license, and notice files are retained inside each component.

| Directory | Source license | Source-release status |
| --- | --- | --- |
| `GeoCalib/` | Apache-2.0 | Bundled; retain the included license and notices |
| `MoGe/` | MIT, with bundled Apache-2.0 components | Bundled; retain the included license |
| `mega-sam/` | Apache-2.0 software; CC BY 4.0 other materials | Bundled with disclosed Apache-2.0 modifications |
| `HaWoR/` | CC BY-NC-ND 4.0 | Local only and Git-ignored; separate permission is required for this modified infra copy to be redistributed |

This is a license summary, not legal advice. The local HaWoR copy contains a
Ego2LeRobot-specific checkpoint/MANO path adaptation, so its `NoDerivatives` terms do
not permit this copy to be included in a public source release. It may only be
used internally for non-commercial purposes covered by the license, unless
separate permission is obtained from the HaWoR authors.

Model weights, generated binaries, output-heavy demo notebooks, and separately
licensed assets are intentionally excluded from source releases. This includes
GeoCalib/MoGe/Mega-SAM/HaWoR checkpoints, DROID-SLAM and Metric3D weights, and
MANO files. Users must obtain those files from their official sources and
comply with their separate terms. MANO requires account registration and
acceptance of the official MANO license.

## Local Mega-SAM changes

The infra snapshot contained three long-video memory fixes that differ from the
base upstream import. They are preserved and marked in the corresponding files:

- `mega-sam/base/droid_slam/depth_video.py`
- `mega-sam/base/droid_slam/droid.py`
- `mega-sam/base/droid_slam/trajectory_filler.py`

The changes avoid allocating unused full-resolution disparity, release dead RGB
buffers before full bundle adjustment, and encode trajectory-filler frames in
bounded chunks. They are distributed under Mega-SAM's Apache-2.0 software
license.
