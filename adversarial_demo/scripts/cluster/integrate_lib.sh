#!/usr/bin/env bash
#
# Shared helpers for the presets that fold a NEWLY computed attack into an EXISTING
# full-dataset run (preset_targeted_l2.sh, preset_dtd_blur.sh).
#
# The merge job rebuilds the combined results from whatever shards are on disk, so the
# merge attack list must be "every attack already sharded into RESULTS_BASE, plus the new
# one". Hardcoding that list breaks as soon as another attack is folded in (order
# dependence), so we detect it from the shard directory names instead.

# Attacks that already have shards under <results_base>/shards, space separated.
# Shard dirs are named "<attack>__w<start>_<end>".
detect_shard_attacks() {
  local base="$1"
  [[ -d "${base}/shards" ]] || return 0
  ls -1 "${base}/shards" 2>/dev/null \
    | sed -E 's/__w[0-9]+_[0-9]+$//' \
    | sort -u \
    | tr '\n' ' '
}

# Attacks whose shards are known to carry robustness (JPEG/blur) samples: the original
# full run's robustness scope plus the attacks our presets compute with robustness on.
# merge_shards takes robustness_attack_types EXPLICITLY (it does not auto-detect like the
# other ablations), so passing a subset would silently drop the existing curves from the
# re-merged robustness JSON.
ROBUSTNESS_CAPABLE="${ROBUSTNESS_CAPABLE:-geoshield dtd encoder sampling targeted_l2 dtd_blur}"

# union_lists "a b" "b c"  -> "a b c"  (order preserved, deduped)
union_lists() {
  printf '%s\n' $1 $2 | awk 'NF && !seen[$0]++' | tr '\n' ' '
}

# intersect_lists "a b c" "b c d" -> "b c"
intersect_lists() {
  local x y out=""
  for x in $1; do
    for y in $2; do
      if [[ "$x" == "$y" ]]; then out+="$x "; break; fi
    done
  done
  echo "$out"
}

# Echo the merge + robustness attack lists for folding NEW_ATTACK into RESULTS_BASE.
# Falls back to the known original 8 if the shards dir is missing (nothing detected).
resolve_integration_lists() {
  local base="$1" new_attack="$2"
  local detected fallback="encoder sampling diffusion_l2 dtd ace geoshield training_loss unidef"
  detected="$(detect_shard_attacks "${base}")"
  [[ -z "${detected// /}" ]] && detected="${fallback}"
  INTEGRATION_MERGE_ATTACKS="$(union_lists "${detected}" "${new_attack}")"
  INTEGRATION_ROBUSTNESS_ATTACKS="$(union_lists \
      "$(intersect_lists "${INTEGRATION_MERGE_ATTACKS}" "${ROBUSTNESS_CAPABLE}")" \
      "${new_attack}")"
  return 0
}
