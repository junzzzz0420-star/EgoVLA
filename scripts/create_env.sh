#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYPI_INDEX_URL="${EGO2LEROBOT_PYPI_INDEX_URL:-https://mirrors.ustc.edu.cn/pypi/simple/}"
NUMPY_INDEX_URL="${EGO2LEROBOT_NUMPY_INDEX_URL:-$PYPI_INDEX_URL}"
TORCH_INDEX_URL="${EGO2LEROBOT_TORCH_INDEX_URL:-$PYPI_INDEX_URL}"
CONDA_FALLBACK_CHANNELS="${EGO2LEROBOT_CONDA_FALLBACK_CHANNELS:-https://conda.anaconda.org/conda-forge}"
PIP_NETWORK_ARGS=(--retries 10 --timeout 120)
PYPI_SOURCE_ARGS=(--index-url "$PYPI_INDEX_URL")
TORCH_REQUIREMENTS="$PROJECT_DIR/environments/requirements-torch.txt"
DATA_REQUIREMENTS="$PROJECT_DIR/environments/requirements-data.txt"
ENV_FILE="$PROJECT_DIR/environments/ego2lerobot.yml"
ENV_NAME="${EGO2LEROBOT_ENV_NAME:-ego2lerobot}"
TORCH_SOURCE_ARGS=(--index-url "$TORCH_INDEX_URL")
if [[ -n "${EGO2LEROBOT_TORCH_FIND_LINKS:-}" ]]; then
  TORCH_SOURCE_ARGS=(--index-url "$PYPI_INDEX_URL" --find-links "$EGO2LEROBOT_TORCH_FIND_LINKS")
fi

if [[ "${1:-}" != "" && "${1:-}" != "data" ]]; then
  echo "Usage: $0 [data]" >&2
  exit 2
fi

ENV_PREFIX="$(conda info --json | python -c '
import json
import os
import sys

info = json.load(sys.stdin)
print(os.path.join(info["envs_dirs"][0], sys.argv[1]))
' "$ENV_NAME")"
NUMPY_SPEC="$(sed -n '/^numpy[<>=!~]/ { p; q; }' "$DATA_REQUIREMENTS")"
if [[ -z "$NUMPY_SPEC" ]]; then
  echo "No NumPy requirement was found in requirements-data.txt." >&2
  exit 1
fi

TEMP_CONDA_SPECS=""

cleanup_temp_specs() {
  if [[ -n "$TEMP_CONDA_SPECS" && -f "$TEMP_CONDA_SPECS" ]]; then
    find "$TEMP_CONDA_SPECS" -delete
  fi
}

trap cleanup_temp_specs EXIT

create_environment() {
  local -a create_args=(--verbose --prefix "$ENV_PREFIX" --file "$ENV_FILE")
  local -a fallback_channels=()
  local channel
  local system_channel_config

  if [[ -e "$ENV_PREFIX" ]]; then
    create_args=(--verbose --force --prefix "$ENV_PREFIX" --file "$ENV_FILE")
  fi

  system_channel_config="$(
    conda config --show channels channel_alias default_channels custom_channels 2>/dev/null || true
  )"
  if grep -Eiq 'tuna\.tsinghua\.edu\.cn|mirrors\.hit\.edu\.cn' \
    <<< "$system_channel_config"; then
    echo "[1/4] Skipping system Conda channels because a blocked TUNA or HIT source is configured"
  else
    echo "[1/4] Creating '$ENV_NAME' with the system Conda channels (Conda progress follows)"
    echo "      First-time metadata retrieval and dependency solving may take several minutes."
    echo "      If Conda stays at 'Collecting package metadata (repodata.json)', it is still working."
    if conda env create "${create_args[@]}"; then
      return 0
    fi
  fi

  read -r -a fallback_channels <<< "$CONDA_FALLBACK_CHANNELS"
  if (( ${#fallback_channels[@]} == 0 )); then
    echo "System Conda channels failed and no fallback channel is configured." >&2
    return 1
  fi

  echo >&2
  echo "Trying strictly isolated fallback Conda channels." >&2
  echo "The fallback ignores configured channels and does not modify Conda settings." >&2
  TEMP_CONDA_SPECS="$(mktemp "${TMPDIR:-/tmp}/ego2lerobot-conda-specs.XXXXXX")"
  sed -n 's/^  - //p' "$ENV_FILE" > "$TEMP_CONDA_SPECS"
  if [[ ! -s "$TEMP_CONDA_SPECS" ]]; then
    echo "No Conda package specifications were found in $ENV_FILE." >&2
    return 1
  fi

  for channel in "${fallback_channels[@]}"; do
    echo "Retrying Conda environment creation with: $channel" >&2
    echo "Metadata retrieval and dependency solving may again take several minutes." >&2
    if conda create --yes --verbose --prefix "$ENV_PREFIX" \
      --override-channels --channel "$channel" --strict-channel-priority \
      --no-default-packages --show-channel-urls --file "$TEMP_CONDA_SPECS"; then
      return 0
    fi
  done

  echo >&2
  echo "Environment creation failed with the system and fallback Conda channels." >&2
  echo "Set EGO2LEROBOT_CONDA_FALLBACK_CHANNELS to space-separated reachable channel URLs." >&2
  return 1
}

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[1/4] Reusing existing '$ENV_NAME' environment and resuming package installation"
else
  if [[ -e "$ENV_PREFIX" ]]; then
    echo "[1/4] Replacing an incomplete Conda prefix at the target environment location"
  fi
  create_environment
fi

echo "[2/4] Installing NumPy and PyTorch"
echo "      NumPy source: $NUMPY_INDEX_URL"
conda run --no-capture-output --name "$ENV_NAME" python -m pip install \
  --progress-bar on --only-binary=:all: --no-deps \
  "${PIP_NETWORK_ARGS[@]}" --index-url "$NUMPY_INDEX_URL" "$NUMPY_SPEC"

echo "      PyTorch source: $TORCH_INDEX_URL"
conda run --no-capture-output --name "$ENV_NAME" python -m pip install \
  --progress-bar on \
  "${PIP_NETWORK_ARGS[@]}" "${TORCH_SOURCE_ARGS[@]}" \
  --requirement "$TORCH_REQUIREMENTS"

echo "[3/4] Installing data-pipeline dependencies"
conda run --no-capture-output --name "$ENV_NAME" python -m pip install \
  --progress-bar on \
  "${PIP_NETWORK_ARGS[@]}" "${PYPI_SOURCE_ARGS[@]}" \
  --constraint "$TORCH_REQUIREMENTS" \
  --requirement "$DATA_REQUIREMENTS"

echo "[4/4] Installing the package in editable mode"
conda run --no-capture-output --name "$ENV_NAME" python -m pip install \
  --progress-bar on --no-deps --editable "$PROJECT_DIR"

conda run --no-capture-output --cwd "$PROJECT_DIR" --name "$ENV_NAME" \
  python -m cli doctor --profile data --strict

echo
echo "Environment '$ENV_NAME' is ready."
echo "Activate it with: conda activate $ENV_NAME"
