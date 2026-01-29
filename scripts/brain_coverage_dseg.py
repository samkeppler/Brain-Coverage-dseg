#!/usr/bin/env python3
# =============================================================================
# Purpose: Calculate brain coverage for each subject’s DWI data using ACPC-space
#          masks (whole brain + dseg region masks) and output results to QC dir.
#          Outputs participant_id + coverage per region (no sessions).
# Adapted on 1/22/26 by Samantha Keppler
# =============================================================================

import os
import shutil
from glob import glob
from datetime import datetime
from typing import Optional

import pandas as pd
import nibabel as nib
from nipype.interfaces import fsl
from nipype.interfaces.fsl.maths import MathsCommand

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "base_path": "/mnt/synapse/neurocat-lab/R21MH133229_asd_dmri_lifespan/datasets_v1.0",
    "dataset_name": "abideii-ip",

    # Deterministic identifier for the intended DWI series (no wildcard).
    # Example: "dir-PA_space-ACPC" or "dir-PA_run-01_space-ACPC"
    # Can be overridden at runtime with environment variable DWI_PREFIX
    "dwi_prefix": "dir-PA_space-ACPC",

    "paths": {
        "qsiprep_root": "{base}/{dataset}/derivatives/qsiprep-1.0.0rc2",
        "output_dir": "{base}/{dataset}/derivatives/qsiprep-1.0.0rc2/qc",
        "output_tsv": "{base}/{dataset}/derivatives/qsiprep-1.0.0rc2/qc/brain_dseg_masks_coverage.tsv",
    },

    # Brainstem intentionally excluded per advisor guidance
    "mask_templates": {
        "icbm152": "{qsiprep_root}/{subj}/dwi/{subj}_space-ACPC_mni_icbm152_brain_coverage_mask.nii.gz",
        "cerebrum": "{qsiprep_root}/{subj}/dwi/{subj}_space-ACPC_mni_cerebrum_brain_coverage_mask.nii.gz",
        "cerebellum_and_midbrain": "{qsiprep_root}/{subj}/dwi/{subj}_space-ACPC_mni_cerebellum_and_midbrain_brain_coverage_mask.nii.gz",
    },

    "options": {
        # If deterministic dwi (built from dwi_prefix) is missing, allow trying a wildcard fallback.
        # Set False to enforce strict deterministic behavior.
        "allow_wildcard_fallback": True,
        "keep_intermediates": False,
        "verbose": True,
    }
}

# =============================================================================
# HELPERS
# =============================================================================

def count_nonzero_voxels(img_path: str) -> float:
    data = nib.load(img_path).get_fdata()
    return float((data != 0).sum())


def get_subject_list(qsiprep_root: str):
    subs = sorted(
        os.path.basename(p)
        for p in glob(os.path.join(qsiprep_root, "sub-*"))
        if os.path.isdir(p)
    )
    if not subs:
        raise RuntimeError(f"No subject folders found under: {qsiprep_root}")
    return subs


def find_preproc_dwi(subj: str, qsiprep_root: str, dwi_prefix: str, allow_fallback: bool) -> Optional[str]:
    """
    Primary behavior: deterministic filename using dwi_prefix (no wildcard).
    If that file does not exist and allow_fallback is True, try a glob wildcard
    as a fallback and pick the first match (with a warning).
    """
    expected = os.path.join(qsiprep_root, subj, "dwi", f"{subj}_{dwi_prefix}_desc-preproc_dwi.nii.gz")
    if os.path.exists(expected):
        return expected

    # deterministic file missing
    if not allow_fallback:
        return None

    # Try a safe wildcard fallback (only within the subject's dwi folder)
    pattern = os.path.join(qsiprep_root, subj, "dwi", f"{subj}_*_desc-preproc_dwi.nii.gz")
    candidates = sorted(glob(pattern))
    if candidates:
        # warn the user and return the first candidate
        print(f"  [WARN] Deterministic DWI not found for {subj} (expected: {os.path.basename(expected)}).")
        print(f"         Falling back to first match: {os.path.basename(candidates[0])}")
        return candidates[0]
    # nothing found
    return None


def compute_coverage(subj: str, dwi_file: str, mask_file: str, work_dir: str) -> Optional[float]:
    subj_tmp = os.path.join(work_dir, subj)
    os.makedirs(subj_tmp, exist_ok=True)

    dwi_float = os.path.join(subj_tmp, f"{subj}_dwi_float.nii.gz")
    dwi_mean = os.path.join(subj_tmp, f"{subj}_dwi_meanT.nii.gz")
    dwi_mean_bin = os.path.join(subj_tmp, f"{subj}_dwi_meanT_bin.nii.gz")
    masked = os.path.join(subj_tmp, f"{subj}_masked.nii.gz")

    MathsCommand(
        in_file=dwi_file,
        out_file=dwi_float,
        output_datatype="float",
        output_type="NIFTI_GZ"
    ).run()

    if not os.path.exists(dwi_float):
        return None

    fsl.MeanImage(
        in_file=dwi_float,
        out_file=dwi_mean,
        dimension="T",
        output_type="NIFTI_GZ"
    ).run()

    fsl.UnaryMaths(
        in_file=dwi_mean,
        out_file=dwi_mean_bin,
        operation="bin",
        output_type="NIFTI_GZ"
    ).run()

    fsl.ApplyMask(
        in_file=dwi_mean_bin,
        mask_file=mask_file,
        out_file=masked,
        output_type="NIFTI_GZ"
    ).run()

    if not os.path.exists(masked):
        return None

    n_mask = count_nonzero_voxels(mask_file)
    if n_mask == 0:
        return None

    n_cov = count_nonzero_voxels(masked)
    return round((n_cov / n_mask) * 100.0, 3)


# =============================================================================
# MAIN
# =============================================================================

def main():
    base = CONFIG["base_path"]
    dataset = CONFIG["dataset_name"]

    # allow environment override for quick tests / different runs
    env_prefix = os.environ.get("DWI_PREFIX")
    dwi_prefix = env_prefix if env_prefix is not None else CONFIG["dwi_prefix"]

    qsiprep_root = CONFIG["paths"]["qsiprep_root"].format(base=base, dataset=dataset)
    output_dir = CONFIG["paths"]["output_dir"].format(base=base, dataset=dataset)
    out_tsv = CONFIG["paths"]["output_tsv"].format(base=base, dataset=dataset)

    os.makedirs(output_dir, exist_ok=True)
    work_dir = os.path.join(output_dir, "tmp")
    os.makedirs(work_dir, exist_ok=True)

    mask_templates = {
        k: v.format(qsiprep_root=qsiprep_root, subj="{subj}")
        for k, v in CONFIG["mask_templates"].items()
    }

    subjects = get_subject_list(qsiprep_root)
    total = len(subjects)

    start = datetime.now()
    print(f"Started at {start}")
    print(f"Using DWI prefix: {dwi_prefix} (allow fallback: {CONFIG['options']['allow_wildcard_fallback']})")

    rows = []
    for i, subj in enumerate(subjects, 1):
        print(f"[{i}/{total}] Processing {subj} ...", flush=True)

        dwi_file = find_preproc_dwi(subj, qsiprep_root, dwi_prefix, CONFIG["options"]["allow_wildcard_fallback"])

        row = {"participant_id": subj}
        if dwi_file is None:
            if CONFIG["options"]["verbose"]:
                print(f"  [WARN] Missing DWI for {subj} (expected {subj}_{dwi_prefix}_desc-preproc_dwi.nii.gz)")
            for name in mask_templates:
                row[f"brain_coverage_{name}"] = None
            rows.append(row)
            continue

        for name, tmpl in mask_templates.items():
            mask_file = tmpl.format(subj=subj)
            if not os.path.exists(mask_file):
                if CONFIG["options"]["verbose"]:
                    print(f"  [WARN] Missing mask for {subj}: {mask_file}")
                row[f"brain_coverage_{name}"] = None
                continue

            row[f"brain_coverage_{name}"] = compute_coverage(
                subj, dwi_file, mask_file, work_dir
            )

        rows.append(row)

        if not CONFIG["options"]["keep_intermediates"]:
            shutil.rmtree(os.path.join(work_dir, subj), ignore_errors=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_tsv, sep="\t", index=False)

    print(f"\nSaved QC results to: {out_tsv}")
    print(f"Total runtime: {datetime.now() - start}")


if __name__ == "__main__":
    main()
