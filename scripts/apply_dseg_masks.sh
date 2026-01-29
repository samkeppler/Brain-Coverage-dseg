#!/bin/bash
# =============================================================================
# Purpose: Apply QSIPrep MNI152NLin2009cAsym -> subject ACPC transform to multiple
#          MNI-space binary/label masks and resample them into each subject's
#          ACPC-space DWI reference grid (dwiref). The resulting subject-space
#          masks are prerequisites for region-wise brain coverage metrics
#          (e.g., cerebrum, cerebellum, brainstem).
#
# Created on 2026-01-29 by Samantha Keppler
#
# Requirements:
# - Docker
# - antsx/ants:2.5.3 container image (pulled automatically by docker run)
# - QSIPrep outputs containing:
#   - sub-<ID>/anat/sub-<ID>_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5
#   - sub-<ID>/dwi/sub-<ID>_dir-*_space-ACPC_dwiref.nii.gz
#
# Notes:
# - Uses NearestNeighbor interpolation for binary/label masks.
# - Skips work if inputs are missing or outputs already exist.
# =============================================================================
set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG=(
  # Dataset paths
  ["bids_dir"]="/mnt/synapse/neurocat-lab/R21MH133229_asd_dmri_lifespan/datasets_v1.0/abideii-ip"
  ["qsiprep_version"]="qsiprep-1.0.0rc2"
  ["sj_list_file"]="/mnt/synapse/neurocat-lab/R21MH133229_asd_dmri_lifespan/datasets_v1.0/abideii-ip/code/sj_list.txt"

  # Mask inputs (MNI space)
  ["mni_masks_dir"]="/mnt/synapse/neurocat-lab/atlases/MNI152NLin2009cAsym_dseg_wb_masks"

  # Transform name (QSIPrep composite transform: MNI152NLin2009cAsym -> ACPC)
  ["xfm_name_template"]="sub-{subj}_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5"

  # Docker image containing antsApplyTransforms
  ["ants_docker_image"]="antsx/ants:2.5.3"

  # Interpolation for label/binary masks
  ["interp"]="NearestNeighbor"
)

# Masks (filenames must exist under CONFIG[mni_masks_dir])
MASKS=(
  "cerebrum.nii.gz"
  "cerebellum_and_midbrain.nii.gz"
  "brainstem.nii.gz"
)

# Output tags (used in output filenames); key is mask stem without .nii.gz
declare -A OUTTAG=(
  ["cerebrum"]="mni_cerebrum_brain_coverage_mask"
  ["cerebellum_and_midbrain"]="mni_cerebellum_and_midbrain_brain_coverage_mask"
  ["brainstem"]="mni_brainstem_brain_coverage_mask"
)

# =============================================================================
# FUNCTIONS
# =============================================================================

die () {
  echo "ERROR: $*" 1>&2
  exit 1
}

qsiprep_dir () {
  echo "${CONFIG[bids_dir]}/derivatives/${CONFIG[qsiprep_version]}"
}

apply_xfm_mni2acpc_mask_docker () {
  local qsiprep_dir_path="$1"
  local subj="$2"
  local in_file="$3"
  local out_file="$4"
  local ref_file="$5"
  local interp="$6"

  # Existence checks
  if [[ ! -f "$in_file" ]]; then
    echo "Skipping - input file does not exist: $in_file"
    return 0
  fi
  if [[ ! -f "$ref_file" ]]; then
    echo "Skipping - reference file does not exist: $ref_file"
    return 0
  fi
  if [[ -f "$out_file" ]]; then
    echo "Skipping - output file already exists: $out_file"
    return 0
  fi

  # QSIPrep composite transform (MNI -> ACPC)
  local xfm_dir="${qsiprep_dir_path}/sub-${subj}/anat"
  local xfm_name="${CONFIG[xfm_name_template]//\{subj\}/$subj}"
  local xfm_path="${xfm_dir}/${xfm_name}"

  if [[ ! -f "$xfm_path" ]]; then
    echo "Skipping - transform file does not exist: $xfm_path"
    return 0
  fi

  echo "---------------------------------------------"
  echo "qsiprep dir:    $qsiprep_dir_path"
  echo "subject:        sub-${subj}"
  echo "input file:     $in_file"
  echo "output file:    $out_file"
  echo "reference file: $ref_file"
  echo "transform:      $xfm_path"
  echo "interp:         $interp"

  # Docker mount points
  local in_dir out_dir ref_dir
  in_dir="$(dirname "$in_file")"
  out_dir="$(dirname "$out_file")"
  ref_dir="$(dirname "$ref_file")"
  mkdir -p "$out_dir"

  local in_base out_base ref_base
  in_base="$(basename "$in_file")"
  out_base="$(basename "$out_file")"
  ref_base="$(basename "$ref_file")"

  # Run ANTs in container
  docker run --rm \
    -v "$in_dir":/input:ro \
    -v "$out_dir":/output \
    -v "$xfm_dir":/xfm:ro \
    -v "$ref_dir":/ref:ro \
    "${CONFIG[ants_docker_image]}" \
    antsApplyTransforms \
      -i "/input/$in_base" \
      -t "/xfm/$xfm_name" \
      -r "/ref/$ref_base" \
      -o "/output/$out_base" \
      -n "$interp"

  echo "Wrote: $out_file"
}

find_dwiref () {
  local qsiprep_dir_path="$1"
  local subj="$2"
  local ref_glob="${qsiprep_dir_path}/sub-${subj}/dwi/sub-${subj}_dir-*_space-ACPC_dwiref.nii.gz"

  # Robust to dir-PA vs dir-AP etc.; pick first match
  ls -1 $ref_glob 2>/dev/null | head -n 1 || true
}

# =============================================================================
# MAIN
# =============================================================================

main () {
  command -v docker >/dev/null 2>&1 || die "docker not found in PATH"

  local qsiprep_dir_path
  qsiprep_dir_path="$(qsiprep_dir)"

  [[ -d "${CONFIG[bids_dir]}" ]] || die "bids_dir not found: ${CONFIG[bids_dir]}"
  [[ -d "$qsiprep_dir_path" ]] || die "qsiprep_dir not found: $qsiprep_dir_path"
  [[ -f "${CONFIG[sj_list_file]}" ]] || die "subject list not found: ${CONFIG[sj_list_file]}"
  [[ -d "${CONFIG[mni_masks_dir]}" ]] || die "mni_masks_dir not found: ${CONFIG[mni_masks_dir]}"

  while read -r subj; do
    [[ -z "$subj" ]] && continue

    local ref_file
    ref_file="$(find_dwiref "$qsiprep_dir_path" "$subj")"
    if [[ -z "$ref_file" ]]; then
      echo "Skipping - could not find dwiref for sub-${subj} in ${qsiprep_dir_path}/sub-${subj}/dwi/"
      continue
    fi

    for mask_fn in "${MASKS[@]}"; do
      local in_file stem tag out_file
      in_file="${CONFIG[mni_masks_dir]}/${mask_fn}"

      stem="${mask_fn%.nii.gz}"  # cerebrum, cerebellum_and_midbrain, brainstem
      tag="${OUTTAG[$stem]:-mni_${stem}_brain_coverage_mask}"

      out_file="${qsiprep_dir_path}/sub-${subj}/dwi/sub-${subj}_space-ACPC_${tag}.nii.gz"

      apply_xfm_mni2acpc_mask_docker \
        "$qsiprep_dir_path" \
        "$subj" \
        "$in_file" \
        "$out_file" \
        "$ref_file" \
        "${CONFIG[interp]}"
    done

  done < "${CONFIG[sj_list_file]}"
}

main "$@"
