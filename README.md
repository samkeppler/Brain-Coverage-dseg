# Brain Coverage QC for Diffusion MRI
<<<<<<< HEAD

## About

Brain-Coverage-dseg is an automated tool for quantifying brain coverage in MRI data. It calculates coverage metrics for the whole brain, cerebrum, and cerebellum using anatomically defined regional masks. The tool was developed and validated for diffusion MRI data preprocessed using QSIPrep.

> Portions of the coverage-calculation scripts are adapted from DCAN-Labs' `brain_coverage` project — see [ATTRIBUTION.md](ATTRIBUTION.md) for the source and full list of modifications.

## Installation

**Requirements:**

- Python 3 with `pandas`, `nibabel`, and `nipype`
- FSL (accessed via `nipype.interfaces.fsl`)
- Docker (used to run `antsApplyTransforms` via the `antsx/ants:2.5.3` image)

**Steps:**

1. Clone this repository.
2. Install the required Python packages (`pandas`, `nibabel`, `nipype`).
3. Install FSL and confirm it's available on your system path.
4. Install Docker and pull the `antsx/ants:2.5.3` image.

## Input Data Requirements

Input diffusion MRI data must be processed with QSIPrep before running Brain-Coverage-dseg.

## How to Run

Brain-Coverage-dseg runs in two steps. Use the `_sessions` version of each script if your dataset has multiple sessions per subject; otherwise use the single-session version.

1. **Transform the region masks into each subject's diffusion space:**
   ```
   ./apply_dseg_masks.sh
   ```
   or, for multi-session datasets:
   ```
   ./apply_dseg_masks_sessions.sh
   ```
   This aligns the whole-brain, cerebrum, and cerebellum masks to each subject's native diffusion space, using their QSIPrep outputs.

2. **Calculate brain coverage:**
   ```
   python brain_coverage_dseg.py
   ```
   or, for multi-session datasets:
   ```
   python brain_coverage_dseg_sessions.py
   ```
   This calculates, for each subject (and session), what percentage of each region is covered by usable diffusion data.

   Note: the expected DWI filename prefix can be changed via the `DWI_PREFIX` environment variable, if needed.

## Outputs

- **Single-session datasets:** one CSV file, `brain_coverage/results/brain_coverage_dseg_masks.csv`, with columns `participant_id`, `coverage_icbm152`, `coverage_superior_cerebrum`, `coverage_inferior_cerebrum`, and `coverage_cerebellum_and_midbrain`.
- **Multi-session datasets:** one CSV file per session (e.g., `brain_coverage_dseg_masks_ses-1.csv`, `..._ses-2.csv`), with the same columns.
