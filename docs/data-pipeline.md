# Data pipeline

The Ray pipeline reconstructs camera and bimanual MANO trajectories from
filtered egocentric video, then exports LeRobot v3. Convert that export to the
14-D Pi0.5 training format with `ego_pipeline/tasks/lerobot_to_pi05.py`.

See [`README.md`](../README.md) for the production commands.

## Local requirements

1. Read `THIRD_PARTY_NOTICES.md` and verify that the intended use is permitted.
2. Obtain GeoCalib, MoGe, Mega-SAM, and HaWoR under their original license terms.
3. Obtain the required checkpoints and MANO files from their official sources.
4. Validate the environment:

```bash
python -m cli doctor --profile data --strict
```

## Pipeline

```bash
python -m cli pipeline \
  --input /path/to/filtered-videos \
  --output /path/to/pipeline-out \
  --num_gpus 8 \
  --per_video_lerobot \
  --aggregate_lerobot
```

Resume the same output instead of copying partial files:

```bash
python -m cli pipeline \
  --input /path/to/filtered-videos \
  --resume /path/to/pipeline-out \
  --num_gpus 8 \
  --retry_failed
```

## Output contract

- per-video prediction and cleaned trajectory files
- sidecar metadata that uses paths relative to the run root
- LeRobot v3 Parquet and video chunks
- a resumable manifest with status and failure summaries
