# Brain Coverage QC for Diffusion MRI

This repository implements a two-stage workflow for computing brain
coverage metrics from qsiprep-preprocessed diffusion MRI data.

> Portions of the coverage-calculation scripts are adapted from DCAN-Labs'
> `brain_coverage` project — see [ATTRIBUTION.md](ATTRIBUTION.md) for
> the source and full list of modifications.

## Conceptual Overview

First, a preprocessing script applies MNI-space anatomical masks (ICBM152,
superior cerebrum, inferior cerebrum, and cerebellum + midbrain) to each
subject's native ACPC diffusion space, producing subject-specific,
region-level masks aligned to the diffusion data.

Second, a coverage computation script uses those subject-specific masks to
quantify what proportion of each region is covered by the diffusion data.
Coverage is computed by thresholding the mean diffusion image, applying the
region mask, and calculating the percentage of nonzero voxels relative to
the full mask extent.

## Scripts

All scripts live in `scripts/`. Every script has a session-friendly
`_sessions` counterpart, which auto-discovers subjects/sessions from the
qsiprep directory structure and writes one output CSV per session. Use
whichever variant matches your dataset's directory structure.

| Script | Dataset structure | Stage |
|---|---|---|
| `apply_dseg_masks.sh` | No sessions (`sub-<ID>/`) | 1: mask transformation |
| `apply_dseg_masks_sessions.sh` | With sessions (`sub-<ID>/ses-<N>/`) | 1: mask transformation |
| `brain_coverage_dseg.py` | No sessions | 2: coverage calculation |
| `brain_coverage_dseg_sessions.py` | With sessions | 2: coverage calculation |

## Mask Transformation Preprocessing

`apply_dseg_masks.sh` / `apply_dseg_masks_sessions.sh` transform a set of
anatomical masks defined in MNI152NLin2009cAsym space into each subject's
native ACPC diffusion space. This is a required preprocessing step for the
coverage calculation below.

### What the scripts do

For each subject (and, in the sessions variant, each session), the script:

1. Locates the QSIPrep-provided composite transform from
   MNI152NLin2009cAsym space to subject ACPC space.
2. Identifies the subject's diffusion reference image
   (`*_space-ACPC_dwiref.nii.gz`) to define the target grid.
3. Applies the MNI→ACPC transform to each input mask using
   `antsApplyTransforms` (via the `antsx/ants:2.5.3` Docker container).
4. Resamples each mask into the subject's diffusion reference space using
   nearest-neighbor interpolation to preserve label integrity.
5. Writes subject-specific, ACPC-space mask files into the
   `brain_coverage/` output directory.

This is repeated for each of the four anatomical masks (ICBM152, superior
cerebrum, inferior cerebrum, and cerebellum + midbrain).

### Inputs

- A list of subject identifiers (`sj_list.txt`) — single-session variant
  only; the sessions variant discovers subjects and sessions directly from
  the qsiprep output directory structure
- MNI-space anatomical mask templates, stored in the `masks/` directory of
  this repository
- QSIPrep derivatives containing the MNI→ACPC composite transform and
  ACPC-space diffusion reference image for each subject

### Outputs

For each subject (and session) and each mask, the script generates an
ACPC-space mask with a deterministic filename of the form:

```
sub-<ID>_space-ACPC_<region>_brain_coverage_mask.nii.gz
```

where `<region>` is one of: `mni_icbm152`, `mni_superior_cerebrum`,
`mni_inferior_cerebrum`, `mni_cerebellum_and_midbrain`.

In the sessions variant, these are written under
`brain_coverage/sub-<ID>/ses-<N>/masks/`; in the single-session variant,
under `brain_coverage/sub-<ID>/masks/`. These outputs are used directly by
the coverage computation scripts.

### Notes

- Existing output files are overwritten on rerun.
- Nearest-neighbor interpolation is used to avoid partial-volume artifacts
  in binary or label masks.
- The scripts assume QSIPrep-compliant directory and filename conventions.
- Requires Docker (for `antsApplyTransforms`).

## Brain Coverage Calculation

`brain_coverage_dseg.py` / `brain_coverage_dseg_sessions.py` compute, for
each subject (and session), the percentage of each region mask covered by
nonzero voxels in the subject's mean preprocessed diffusion image.

### What the scripts do

For each subject (and, in the sessions variant, each session), the script:

1. Locates the qsiprep-preprocessed DWI file
   (`*_space-ACPC_desc-preproc_dwi.nii.gz`, with wildcard fallback if the
   deterministic filename isn't found).
2. Converts the DWI to float, computes its mean image over time, and
   binarizes it.
3. Applies each of the four ACPC-space region masks (produced by the
   corresponding `apply_dseg_masks*.sh` script) to the binarized mean
   image.
4. Calculates coverage as the percentage of nonzero voxels in the masked
   image relative to the full extent of the region mask.

### Outputs

- **Single-session variant**: one CSV,
  `brain_coverage/results/brain_coverage_dseg_masks.csv`, with columns
  `participant_id`, `coverage_icbm152`, `coverage_superior_cerebrum`,
  `coverage_inferior_cerebrum`, `coverage_cerebellum_and_midbrain`.
- **Sessions variant**: one CSV per session,
  `brain_coverage/results/brain_coverage_dseg_masks_ses-1.csv`,
  `..._ses-2.csv`, etc., with the same columns.

### Requirements

- Python 3 with `pandas`, `nibabel`, and `nipype`
- FSL (accessed via `nipype.interfaces.fsl`)

### Notes

- Intermediate files are written to a per-dataset temporary working
  directory and removed after each subject unless `keep_intermediates` is
  set to `True` in `CONFIG`.
- The DWI filename prefix can be overridden via the `DWI_PREFIX`
  environment variable without editing the script.

## Citation

If you use this tool, please cite:

> [Paper citation — TBD]

See also the [validation analysis repository](https://github.com/samkeppler/Brain-Coverage-dseg-validation)
for the code used to validate this metric against expert manual QC
ratings.
