#!/bin/bash
set -euo pipefail

# =============================================================================
# Purpose:
#   Transform atlas-derived MNI152NLin2009cAsym region masks into each subject's
#   ACPC-space diffusion reference (dwiref) using the QSIPrep-generated
#   MNI→ACPC composite transform.
#
# =============================================================================
# CONFIGURATION
# =============================================================================

declare -A CONFIG
CONFIG=(
  # QSIPrep derivatives roo
  ["qsiprep_root"]="/mnt/synapse/neurocat-lab/R21MH133229_asd_dmri_lifespan/datasets_v1.0/nda-collection9/derivatives/qsiprep-1.0.0rc2"

  # Directory containing the dseg-derived masks (MNI space)
  ["mni_masks_dir"]="/mnt/synapse/neurocat-lab/atlases/MNI152NLin2009cAsym_res-01_dseg_masks"

  # icbm152 mask (single file)
  ["icbm152_mask_file"]="/mnt/synapse/neurocat-lab/atlases/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c_mask.nii"

  # Docker image containing antsApplyTransforms
  ["ants_docker_image"]="antsx/ants:2.5.3"

  # Interpolation for label/binary masks
  ["interp"]="NearestNeighbor"
)

# =============================================================================
# MASK DEFINITIONS
# =============================================================================

MASKS=(
  "MNI152NLin2009cAsym_upper_cerebrum.nii.gz"
  "MNI152NLin2009cAsym_lower_cerebrum.nii.gz"
  "MNI152NLin2009cAsym_cerebrum.nii.gz"
  "MNI152NLin2009cAsym_cerebellum+midbrain.nii.gz"
  "__ICBM152__"
)

declare -A OUTTAG=(
  ["MNI152NLin2009cAsym_upper_cerebrum"]="mni_upper_cerebrum_brain_coverage_mask"
  ["MNI152NLin2009cAsym_lower_cerebrum"]="mni_lower_cerebrum_brain_coverage_mask"
  ["MNI152NLin2009cAsym_cerebrum"]="mni_cerebrum_brain_coverage_mask"
  ["MNI152NLin2009cAsym_cerebellum+midbrain"]="mni_cerebellum_and_midbrain_brain_coverage_mask"
  ["__ICBM152__"]="mni_icbm152_brain_coverage_mask"
)

# =============================================================================
# HELPERS
# =============================================================================

die () { echo "ERROR: $*" 1>&2; exit 1; }

# Return all sessions for subject (e.g., "ses-1 ses-2"), or "" if none.
list_sessions () {
  local qsiprep_root="$1"
  local subj="$2"
  local hit
  hit=$(ls -d "${qsiprep_root}/sub-${subj}"/ses-* 2>/dev/null || true)
  if [[ -z "$hit" ]]; then
    echo ""
    return 0
  fi
  for d in $hit; do basename "$d"; done
}

# Prefer ses-1 if present, else first ses-*
default_session () {
  local qsiprep_root="$1"
  local subj="$2"
  local s
  s=$(list_sessions "$qsiprep_root" "$subj")
  if [[ -z "$s" ]]; then
    echo ""
    return 0
  fi
  for x in $s; do
    if [[ "$x" == "ses-1" ]]; then
      echo "ses-1"
      return 0
    fi
  done
  echo "$s" | awk '{print $1}'
}

# Find dwiref for subject + session.
find_dwiref () {
  local qsiprep_root="$1"
  local subj="$2"
  local ses="$3"   # "ses-1" / "ses-2" / ""

  local dwi_dir
  if [[ -n "$ses" ]]; then
    dwi_dir="${qsiprep_root}/sub-${subj}/${ses}/dwi"
  else
    dwi_dir="${qsiprep_root}/sub-${subj}/dwi"
  fi

  local prefix
  if [[ -n "$ses" ]]; then
    prefix="sub-${subj}_${ses}_"
  else
    prefix="sub-${subj}_"
  fi

  local hit
  hit=$(ls -1 "${dwi_dir}/${prefix}"dir-*_space-ACPC_dwiref.nii.gz 2>/dev/null | head -n 1 || true)
  if [[ -n "$hit" ]]; then echo "$hit"; return 0; fi

  hit=$(ls -1 "${dwi_dir}/${prefix}"space-ACPC_dwiref.nii.gz 2>/dev/null | head -n 1 || true)
  if [[ -n "$hit" ]]; then echo "$hit"; return 0; fi

  hit=$(ls -1 "${dwi_dir}/${prefix}"*_space-ACPC_dwiref.nii.gz 2>/dev/null | head -n 1 || true)
  echo "$hit"
}

# transform is subject-level (NOT session-level)
find_subject_level_xfm () {
  local qsiprep_root="$1"
  local subj="$2"
  local xfm="${qsiprep_root}/sub-${subj}/anat/sub-${subj}_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5"
  if [[ -f "$xfm" ]]; then
    echo "$xfm"
    return 0
  fi
  echo ""
}

apply_xfm_mni2acpc_mask_docker () {
  local xfm_dir="$1"
  local xfm_name="$2"
  local in_file="$3"
  local out_file="$4"
  local ref_file="$5"
  local interp="$6"

  [[ -f "$in_file" ]]  || { echo "Skipping - missing input: $in_file"; return 0; }
  [[ -f "$ref_file" ]] || { echo "Skipping - missing ref:   $ref_file"; return 0; }

  local xfm_path="${xfm_dir}/${xfm_name}"
  [[ -f "$xfm_path" ]] || { echo "Skipping - missing xfm:   $xfm_path"; return 0; }

  local in_dir out_dir ref_dir
  in_dir="$(dirname "$in_file")"
  out_dir="$(dirname "$out_file")"
  ref_dir="$(dirname "$ref_file")"
  mkdir -p "$out_dir"

  local in_base out_base ref_base
  in_base="$(basename "$in_file")"
  out_base="$(basename "$out_file")"
  ref_base="$(basename "$ref_file")"

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
}

# =============================================================================
# MAIN
# =============================================================================

main () {
  command -v docker >/dev/null 2>&1 || die "docker not found in PATH"

  local qsiprep_root="${CONFIG[qsiprep_root]}"
  [[ -d "$qsiprep_root" ]] || die "qsiprep_root not found: $qsiprep_root"
  [[ -d "${CONFIG[mni_masks_dir]}" ]] || die "mni_masks_dir not found: ${CONFIG[mni_masks_dir]}"
  [[ -f "${CONFIG[icbm152_mask_file]}" ]] || die "icbm152 mask not found: ${CONFIG[icbm152_mask_file]}"

  local braincov_root="${qsiprep_root}/brain_coverage"
  mkdir -p "$braincov_root"

  local subj_dir subj
  for subj_dir in "${qsiprep_root}"/sub-*; do
    [[ -d "$subj_dir" ]] || continue
    subj="$(basename "$subj_dir")"
    subj="${subj#sub-}"

    # Discover all sessions this subject actually has on disk (e.g.
    # "ses-1 ses-2") instead of relying on a static ses-2 subject list file.
    local all_sessions
    all_sessions="$(list_sessions "$qsiprep_root" "$subj")"
    if [[ -z "$all_sessions" ]]; then
      echo "Skipping sub-${subj}: no session folders found"
      continue
    fi

    # De-duplicate (list_sessions shouldn't produce dupes, but stay safe)
    local uniq=()
    for s in $all_sessions; do
      local seen="false"
      for u in "${uniq[@]}"; do [[ "$u" == "$s" ]] && seen="true"; done
      [[ "$seen" == "false" ]] && uniq+=("$s")
    done

    # Subject-level transform (shared across sessions)
    local xfm_path
    xfm_path="$(find_subject_level_xfm "$qsiprep_root" "$subj")"
    if [[ -z "$xfm_path" ]]; then
      echo "Skipping sub-${subj}: missing subject-level xfm: ${qsiprep_root}/sub-${subj}/anat/sub-${subj}_from-MNI152NLin2009cAsym_to-ACPC_mode-image_xfm.h5"
      continue
    fi
    local xfm_dir xfm_name
    xfm_dir="$(dirname "$xfm_path")"
    xfm_name="$(basename "$xfm_path")"

    for ses in "${uniq[@]}"; do
      local ref_file out_mask_dir

      ref_file="$(find_dwiref "$qsiprep_root" "$subj" "$ses")"
      if [[ -z "$ref_file" ]]; then
        echo "Skipping - could not find dwiref for sub-${subj} ${ses}"
        continue
      fi

      out_mask_dir="${braincov_root}/sub-${subj}/${ses}/masks"
      mkdir -p "$out_mask_dir"

      echo "============================================="
      echo "Subject: sub-${subj} ${ses}"
      echo "Output dir:   $out_mask_dir"
      echo "Reference:    $ref_file"
      echo "Transform:    $xfm_path"

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
        out_file="${out_mask_dir}/sub-${subj}_space-ACPC_${tag}.nii.gz"

        # overwrite on rerun
        rm -f "$out_file"

        apply_xfm_mni2acpc_mask_docker \
          "$xfm_dir" \
          "$xfm_name" \
          "$in_file" \
          "$out_file" \
          "$ref_file" \
          "${CONFIG[interp]}"

        echo "  Wrote: $out_file"
      done
    done

  done
}

main "$@"
