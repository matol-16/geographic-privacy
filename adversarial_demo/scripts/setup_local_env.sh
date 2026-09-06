#!/usr/bin/env bash
#
# Reproducible setup of the `plonk` conda env for local (single-GPU) experiments.
# Everything lives under /Data/mathias.ollu (home has little space):
#   - env:        /Data/mathias.ollu/conda/plonk     (per ~/.condarc envs_dirs)
#   - pkg cache:  /Data/mathias.ollu/conda/pkgs      (per ~/.condarc pkgs_dirs)
#   - pip cache:  /Data/mathias.ollu/pip_cache       (kept off $HOME)
#   - tmp:        /Data/mathias.ollu/tmp             (big wheels extract here, not /tmp or $HOME)
#
# Idempotent: re-running it re-installs deps into the existing env. Datasets are handled
# separately by setup_local_data.sh.
#
# Usage:  bash scripts/setup_local_env.sh   (logs to /Data/mathias.ollu/logs/setup_env.log)

set -euo pipefail

ENV_NAME="${ENV_NAME:-plonk}"
PY_VERSION="${PY_VERSION:-3.11}"
DATA_ROOT="/Data/mathias.ollu"
REPO_ROOT="/users/eleves-b/2023/mathias.ollu/repos/plonk"
DEMO_DIR="${REPO_ROOT}/adversarial_demo"
CONDA_BASE="/users/eleves-b/2023/mathias.ollu/miniconda3"

# Keep every large scratch off the space-limited home volume.
export PIP_CACHE_DIR="${DATA_ROOT}/pip_cache"
export TMPDIR="${DATA_ROOT}/tmp"
mkdir -p "${DATA_ROOT}/conda/pkgs" "${PIP_CACHE_DIR}" "${TMPDIR}" "${DATA_ROOT}/logs"

echo "=============================================================="
echo "Setting up conda env '${ENV_NAME}' (python ${PY_VERSION})"
echo "  env root:   ${DATA_ROOT}/conda"
echo "  pip cache:  ${PIP_CACHE_DIR}"
echo "  TMPDIR:     ${TMPDIR}"
echo "=============================================================="

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | grep -qsE "(^|/)${ENV_NAME}[[:space:]]|/conda/${ENV_NAME}$"; then
  echo "Env '${ENV_NAME}' already exists -> updating deps in place."
else
  conda create -y -n "${ENV_NAME}" "python=${PY_VERSION}" pip
fi

conda activate "${ENV_NAME}"
echo "python: $(which python)  ($(python -V 2>&1))"
[[ "$(which python)" == ${DATA_ROOT}/* ]] || { echo "ERROR: env not on ${DATA_ROOT} (got $(which python))"; exit 1; }

python -m pip install --upgrade pip

# Core PLONK package (pulls torch/torchvision/geoopt/... from setup.py; PyPI torch on
# Linux is the CUDA build, which the A5000 driver supports).
python -m pip install -e "${REPO_ROOT}"

# adversarial_demo extras.
python -m pip install -r "${DEMO_DIR}/requirements.txt"

# cartopy: pip builds against system GEOS/PROJ and often fails; conda-forge is robust.
# Only used for map plots, so a failure here is non-fatal for attacks/eval.
conda install -y -c conda-forge cartopy || echo "WARN: cartopy install failed (map plots unavailable; core eval unaffected)"

echo "=============================================================="
echo "Sanity check"
python - <<'PY'
import torch
print("torch      :", torch.__version__)
print("cuda avail :", torch.cuda.is_available())
print("device     :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
import geoopt, transformers, huggingface_hub, matplotlib, reverse_geocoder  # noqa: F401
print("geoopt/transformers/hf_hub/matplotlib/reverse_geocoder: OK")
import plonk  # noqa: F401
print("plonk import: OK")
try:
    import cartopy  # noqa: F401
    print("cartopy: OK")
except Exception as e:
    print("cartopy: MISSING ->", e)
PY
echo "=============================================================="
echo "Done. Activate with:  conda activate ${ENV_NAME}"
