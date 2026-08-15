#!/bin/bash
set -euo pipefail

# =============================================================================
# Purpose:
#   Transform atlas-derived MNI152NLin2009cAsym region masks into each subject's
#   ACPC-space diffusion reference (dwiref) using the QSIPrep-generated
#   MNI->ACPC composite transform.
#
#   Outputs 4 masks per subject:
#     1. cerebellum + midbrain
#     2. ICBM152
#     3. inferior cerebrum
#     4. superior cerebrum
#
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

declare -A CONFIG
CONFIG=(
  # Core identifier
  ["dataset"]="abideii-bni"
  ["qsiprep_version"]="qsiprep-1.0.0rc2"

  # Base paths 
  ["base_dir"]="/mnt/synapse/neurocat-lab/R21MH133229_asd_dmri_lifespan/datasets_v1.0"
  ["atlas_dir"]="/mnt/synapse/neurocat-lab/atlases"

  # Derived paths
  ["bids_dir"]=""              # set after declaration
  ["sj_list_file"]=""          # set after declaration
  ["mni_masks_dir"]=""         # set after declaration
  ["icbm152_mask_file"]=""     # set after declaration

  # Transform + runtime
  ["xfm_name_template"]="sub-{subj}_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5"
  ["ants_docker_image"]="antsx/ants:2.5.3"
  ["interp"]="NearestNeighbor"
)

# Resolve derived paths
CONFIG["bids_dir"]="${CONFIG[base_dir]}/${CONFIG[dataset]}"
CONFIG["sj_list_file"]="${CONFIG[bids_dir]}/code/sj_list.txt"
CONFIG["mni_masks_dir"]="${CONFIG[atlas_dir]}/MNI152NLin2009cAsym_res-01_dseg_masks"
CONFIG["icbm152_mask_file"]="${CONFIG[atlas_dir]}/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii"

# =============================================================================
# MASK DEFINITIONS
# =============================================================================

MASKS=(
  "MNI152NLin2009cAsym_cerebellum+midbrain.nii.gz"
  "__ICBM152__"
  "MNI152NLin2009cAsym_inferior_cerebrum.nii.gz"
  "MNI152NLin2009cAsym_superior_cerebrum.nii.gz"
)

declare -A OUTTAG=(
  ["MNI152NLin2009cAsym_cerebellum+midbrain"]="mni_cerebellum_and_midbrain_brain_coverage_mask"
  ["__ICBM152__"]="mni_icbm152_brain_coverage_mask"
  ["MNI152NLin2009cAsym_inferior_cerebrum"]="mni_inferior_cerebrum_brain_coverage_mask"
  ["MNI152NLin2009cAsym_superior_cerebrum"]="mni_superior_cerebrum_brain_coverage_mask"
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

  [[ -f "$in_file" ]] || { echo "Skipping - missing input: $in_file"; return 0; }
  [[ -f "$ref_file" ]] || { echo "Skipping - missing reference: $ref_file"; return 0; }
  [[ -f "$out_file" ]] && { echo "Skipping - exists: $out_file"; return 0; }

  local xfm_dir="${qsiprep_dir_path}/sub-${subj}/anat"
  local xfm_name="${CONFIG[xfm_name_template]//\{subj\}/$subj}"
  local xfm_path="${xfm_dir}/${xfm_name}"

  [[ -f "$xfm_path" ]] || { echo "Skipping - missing transform: $xfm_path"; return 0; }

  echo "---------------------------------------------"
  echo "subject: sub-${subj}"
  echo "input:   $in_file"
  echo "output:  $out_file"

  local in_dir out_dir ref_dir
  in_dir="$(dirname "$in_file")"
  out_dir="$(dirname "$out_file")"
  ref_dir="$(dirname "$ref_file")"
  mkdir -p "$out_dir"

  docker run --rm \
    -v "$in_dir":/input:ro \
    -v "$out_dir":/output \
    -v "$xfm_dir":/xfm:ro \
    -v "$ref_dir":/ref:ro \
    "${CONFIG[ants_docker_image]}" \
    antsApplyTransforms \
      -i "/input/$(basename "$in_file")" \
      -t "/xfm/$xfm_name" \
      -r "/ref/$(basename "$ref_file")" \
      -o "/output/$(basename "$out_file")" \
      -n "$interp"

  echo "Wrote: $out_file"
}

find_dwiref () {
  local qsiprep_dir_path="$1"
  local subj="$2"
  local dwi_dir="${qsiprep_dir_path}/sub-${subj}/dwi"

  local hit
  hit=$(ls -1 ${dwi_dir}/sub-${subj}_dir-*_space-ACPC_dwiref.nii.gz 2>/dev/null | head -n 1 || true)
  [[ -n "$hit" ]] && { echo "$hit"; return; }

  hit="${dwi_dir}/sub-${subj}_space-ACPC_dwiref.nii.gz"
  [[ -f "$hit" ]] && { echo "$hit"; return; }

  hit=$(ls -1 ${dwi_dir}/sub-${subj}*_space-ACPC_dwiref.nii.gz 2>/dev/null | head -n 1 || true)
  [[ -n "$hit" ]] && { echo "$hit"; return; }

  echo ""
}

# =============================================================================
# MAIN
# =============================================================================

main () {
  command -v docker >/dev/null 2>&1 || die "docker not found"

  local qsiprep_dir_path
  qsiprep_dir_path="$(qsiprep_dir)"

  [[ -d "${CONFIG[bids_dir]}" ]] || die "Missing bids_dir"
  [[ -d "$qsiprep_dir_path" ]] || die "Missing qsiprep_dir"
  [[ -f "${CONFIG[sj_list_file]}" ]] || die "Missing subject list"

  local out_root="${qsiprep_dir_path}/brain_coverage"

  while read -r subj; do
    subj="${subj//[$'\r\t ']/}"
    [[ -z "$subj" ]] && continue
    subj="${subj#sub-}"

    local ref_file
    ref_file="$(find_dwiref "$qsiprep_dir_path" "$subj")"
    [[ -z "$ref_file" ]] && { echo "Skipping sub-${subj} (no dwiref)"; continue; }

    local mask_dir="${out_root}/sub-${subj}/masks"
    mkdir -p "$mask_dir"

    for mask_key in "${MASKS[@]}"; do
      local in_file stem tag out_file

      if [[ "$mask_key" == "__ICBM152__" ]]; then
        in_file="${CONFIG[icbm152_mask_file]}"
        stem="__ICBM152__"
      else
        in_file="${CONFIG[mni_masks_dir]}/${mask_key}"
        stem="${mask_key%.nii.gz}"
      fi

      tag="${OUTTAG[$stem]}"
      out_file="${mask_dir}/sub-${subj}_space-ACPC_${tag}.nii.gz"

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