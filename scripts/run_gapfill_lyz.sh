#!/usr/bin/env bash
# LYZ 双目数据全链测试：run_multiview(stereo, 15s) → gap_fill → audit
set -u
P=/home/jun/miniconda3/envs/ego2lerobot/bin/python
cd /home/jun/Ego2LeRobot
mkdir -p examples/gapfill_lyz
for name in egocap_20261001_060656 egocap_20261001_103655 egocap_20261001_103828 egocap_20261001_104550; do
  exp="examples/egocap_das_export/${name}_bag_0"
  [ -d "$exp" ] || { echo "SKIP(no export) $name"; continue; }
  out="examples/gapfill_lyz/$name"
  echo "===== $name : run_multiview stereo 15s ====="
  "$P" ego_pipeline/preprocessing/das/run_multiview.py --mode stereo \
      --input-export "$exp" --start 2 --duration 15 --out "$out" \
    || { echo "FAIL(run) $name"; continue; }
  f="$out/fused.parquet"
  [ -f "$f" ] || { echo "FAIL(no fused) $name"; continue; }
  echo "----- $name : gap_fill -----"
  "$P" ego_pipeline/preprocessing/das/gap_fill.py --parquet "$f" --max-gap 3 \
    || { echo "FAIL(fill) $name"; continue; }
  echo "----- $name : audit before/after -----"
  "$P" scripts/audit_trajectory.py --parquet "$f" "${f%.parquet}_filled.parquet" \
      --out-dir "$out/qc" | tail -8
done
echo ALL_DONE
