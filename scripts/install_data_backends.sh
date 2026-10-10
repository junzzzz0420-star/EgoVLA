#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$PROJECT_DIR/third_party"
ENV_NAME="${EGO2LEROBOT_ENV_NAME:-ego2lerobot}"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' does not exist. Run scripts/create_env.sh first." >&2
  exit 1
fi

cat <<'NOTICE'
Ego2LeRobot bundles GeoCalib, MoGe, and Mega-SAM source under third_party/. The local
HaWoR source must be supplied separately because this project's adapted copy
cannot be redistributed under HaWoR's NoDerivatives terms. No model weights or
separately licensed assets are bundled.

HaWoR is CC BY-NC-ND 4.0: commercial use and redistribution of modifications
require separate permission. MANO requires a separate license agreement.
Continue only if your use complies with every upstream license.
NOTICE
read -r -p "Type 'I AGREE' to continue: " ACCEPTED
if [[ "$ACCEPTED" != "I AGREE" ]]; then
  echo "Installation cancelled."
  exit 1
fi

require_component() {
  local name="$1"
  local destination="$THIRD_PARTY_DIR/$name"
  if [[ ! -d "$destination" ]]; then
    echo "Bundled source is missing: $destination" >&2
    exit 1
  fi
}

require_component GeoCalib
require_component MoGe
require_component mega-sam
require_component HaWoR

conda run --name "$ENV_NAME" python -m pip install --no-deps --editable "$THIRD_PARTY_DIR/GeoCalib"
conda run --name "$ENV_NAME" python -m pip install --no-deps --editable "$THIRD_PARTY_DIR/MoGe"

# The HaWoR runtime config cannot be distributed (CC BY-NC-ND); link it from the
# locally supplied HaWoR copy instead. detector.pt / hawor.ckpt remain manual
# downloads documented in docs/installation.md.
HAWOR_CFG_SRC="$THIRD_PARTY_DIR/HaWoR/weights/hawor/model_config.yaml"
HAWOR_CFG_DST="$PROJECT_DIR/model/hawor/model_config.yaml"
if [[ -f "$HAWOR_CFG_SRC" && ! -e "$HAWOR_CFG_DST" ]]; then
  ln -s "../../third_party/HaWoR/weights/hawor/model_config.yaml" "$HAWOR_CFG_DST"
  echo "Linked $HAWOR_CFG_DST -> $HAWOR_CFG_SRC"
fi

cat <<'NEXT'

Bundled source backends are registered. Model assets are intentionally absent.
Follow docs/installation.md to place each checkpoint and the licensed MANO
files, then run:

  python -m cli doctor --profile data --strict
NEXT
