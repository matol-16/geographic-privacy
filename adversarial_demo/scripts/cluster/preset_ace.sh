#!/usr/bin/env bash
#
# Fold GeoFA-ACE (attack key `ace`) into a cluster_full run that is ALREADY RUNNING /
# already on disk, reusing the shard + merge infrastructure (submit.sh).
#
#   ./preset_ace.sh yfcc --dry-run     # preview (submits nothing)
#   ./preset_ace.sh yfcc               # 40 GPU jobs -> RESULTS_BASE/shards/ace__w*
#   ./preset_ace.sh osv
#
# WHY A PRESET AND NOT JUST `ATTACK_TYPES_STR=ace ./submit.sh`:
#   * the run's other shards are still being computed, so this must NOT chain a merge
#     (SKIP_MERGE=1) -- merging now would stitch an incomplete run over the combined
#     results. The merge command for the full set is printed at the end instead;
#   * the merge must later stitch the SUPERSET (every attack already sharded into
#     RESULTS_BASE, plus ace), auto-detected by integrate_lib.sh;
#   * robustness scope must be the union, or the re-merged robustness JSON silently
#     drops the curves of the attacks already computed;
#   * ACE is the only TARGETED attack of the run and needs `target_image`; the config's
#     path is relative to PROJECT_DIR, so we pin an absolute one here.
#
# Everything else matches the cluster_full protocol exactly (config.sh's defaults):
# 3 budgets (1/4/8 per 255), 4000 images x 100 per shard, 5 restarts selected by FSD to
# the TRUE location, and all four ablations (sampling steps, cfg, robustness, transfer),
# so ACE lines up point-for-point with the other attacks in every figure and table.
#
# Knobs (all optional):
#   RESULTS_BASE=<dir>    the running run's base (default: config.sh's, i.e. the
#                         cluster_full default for this dataset)
#   AFTER_JOBID=<jobid>   queue the ACE array behind the main array instead of running
#                         alongside it (afterany)
#   MAX_CONCURRENT=<n>    cap concurrent array tasks
#   ABLATION_WINDOWS=<n>  restrict the eval-time ablations to the first n windows (the
#                         main run's cost-control knob -- MATCH WHAT THE RUN USED)
#   TARGET_IMAGE=<path>   ACE's target image (default: <repo>/.media/MIST.png)
#
# Prereqs: the #SBATCH placeholders in eval_shard.slurm / merge.slurm are already filled
# in (the run is going), the plonk conda env exists, DATA_ROOT points at this cluster's
# datasets -- i.e. nothing beyond what the running submission already needed.

set -euo pipefail

DS="${1:?usage: preset_ace.sh <yfcc|osv> [--dry-run]}"; shift || true

export CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${CLUSTER_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
# shellcheck source=/dev/null
source "${CLUSTER_DIR}/integrate_lib.sh"

export DATASET="${DS}"
export ATTACK_TYPES_STR="ace"          # compute ONLY the ace shards

# Resolve RESULTS_BASE the same way config.sh would, so we can detect the sharded
# attacks before sourcing it inside submit.sh.
source_results_base() {
  ( export DATASET="${DS}"; ATTACK_TYPES_STR="ace" source "${CLUSTER_DIR}/config.sh" >/dev/null 2>&1
    echo "${RESULTS_BASE}" )
}
export RESULTS_BASE="${RESULTS_BASE:-$(source_results_base)}"

# ACE is targeted: pin an absolute path to the target image (the config's is relative to
# PROJECT_DIR, which only works when the job's cwd is exactly that dir).
TARGET_IMAGE="${TARGET_IMAGE:-${REPO_ROOT}/.media/MIST.png}"
if [[ ! -f "${TARGET_IMAGE}" ]]; then
  echo "preset_ace.sh: ACE target image not found: ${TARGET_IMAGE}" >&2
  echo "  set TARGET_IMAGE=<path to MIST.png> (it must exist on the compute nodes)." >&2
  exit 1
fi
case "${TARGET_IMAGE}" in *[[:space:]]*)
  echo "preset_ace.sh: TARGET_IMAGE must not contain spaces (it is passed as one --override token)." >&2
  exit 1;;
esac
export EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES_STR:-attack_train_args.${DS}.per_attack.ace.target_image=${TARGET_IMAGE}}"

# Merge the superset (already-sharded attacks + ace); robustness scope = the union.
# Fail loudly if nothing is detected: integrate_lib's fallback is the HISTORICAL 8-attack
# list (no targeted_l2 / sampling_cfg10 / dtd_blur), so a wrong RESULTS_BASE would
# otherwise silently produce a merge that drops three of the run's attacks.
if [[ -z "$(detect_shard_attacks "${RESULTS_BASE}" | tr -d ' ')" && -z "${MERGE_ATTACK_TYPES:-}" ]]; then
  echo "preset_ace.sh: no shards found under ${RESULTS_BASE}/shards" >&2
  echo "  RESULTS_BASE must point at the RUNNING run's base (the dir holding shards/)." >&2
  echo "  Override it, or set MERGE_ATTACK_TYPES=\"...\" explicitly if you know the set." >&2
  exit 1
fi
resolve_integration_lists "${RESULTS_BASE}" ace
export MERGE_ATTACK_TYPES_STR="${MERGE_ATTACK_TYPES:-${INTEGRATION_MERGE_ATTACKS}}"
export ROBUSTNESS_ATTACK_TYPES_STR="${ROBUSTNESS_ATTACK_TYPES:-${INTEGRATION_ROBUSTNESS_ATTACKS}}"

# The rest of the run is still computing -> array only, merge by hand afterwards.
export SKIP_MERGE="${SKIP_MERGE:-1}"

echo "== folding ACE into ${RESULTS_BASE}"
echo "   target image: ${TARGET_IMAGE}"
echo "   merge set:    ${MERGE_ATTACK_TYPES_STR}"
echo "   robustness:   ${ROBUSTNESS_ATTACK_TYPES_STR}"
[[ -n "${AFTER_JOBID:-}" ]] && echo "   queued behind job ${AFTER_JOBID}"

"${CLUSTER_DIR}/submit.sh" "$@"

if [[ " $* " != *" --dry-run "* ]]; then
  cat <<EOF

--------------------------------------------------------------------------------
ACE is queued. It appears in the paper figures/tables automatically once merged:
'ace' is already in make_paper_figures.sh's ALL_ATTACKS and in the success-rate
table's GeoFA family list, and _display_attack_name maps it to "GeoFA-ACE".

After the merge:  ${PROJECT_DIR}/scripts/make_paper_figures.sh ${DS}
                  python ${PROJECT_DIR}/scripts/make_success_rate_table.py
--------------------------------------------------------------------------------
EOF
fi
