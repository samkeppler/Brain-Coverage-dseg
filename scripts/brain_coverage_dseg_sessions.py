#!/usr/bin/env python3
# =============================================================================
# Purpose: Calculate brain coverage for each subject's DWI data using ACPC-space
#          masks and output results under qsiprep/brain_coverage/results.
# Notes  : This version supports datasets with sessions and writes one CSV per
#          session (ses-1 and ses-2). Subjects and their sessions are
#          discovered directly from the qsiprep folder structure on disk.
# =============================================================================

import os
import shutil
from glob import glob
from datetime import datetime
from typing import Optional
from collections import OrderedDict

import pandas as pd
import nibabel as nib
from nipype.interfaces import fsl
from nipype.interfaces.fsl.maths import MathsCommand


CONFIG = {
    "base_path": "/mnt/synapse/neurocat-lab/R21MH133229_asd_dmri_lifespan/datasets_v1.0",
    "dataset_name": "nda-collection9",

    # Sessions to process. Subjects for each session are always discovered by
    # scanning qsiprep_root for sub-*/<ses>/ directories.
    "sessions": ["ses-1", "ses-2"],

    # Can be overridden with environment variable DWI_PREFIX.
    # For this dataset, the DWI file includes the session in its name, e.g.:
    #   sub-XXXX_ses-1_dir-PA_space-ACPC_desc-preproc_dwi.nii.gz
    #
    # We keep dwi_prefix as the non-subject part excluding the leading "{subj}_".
    # The session token is injected automatically (see find_preproc_dwi()).
    "dwi_prefix": "dir-PA_space-ACPC",

    "paths": {
        "qsiprep_root": "{base}/{dataset}/derivatives/qsiprep-1.0.0rc2",

        # Output locations
        "braincov_root": "{base}/{dataset}/derivatives/qsiprep-1.0.0rc2/brain_coverage",
        "results_dir": "{base}/{dataset}/derivatives/qsiprep-1.0.0rc2/brain_coverage/results",

        # We will create one output CSV per session under results_dir
        "output_csv_template": (
            "{base}/{dataset}/derivatives/qsiprep-1.0.0rc2/brain_coverage/results/"
            "brain_coverage_dseg_masks_{ses}.csv"
        ),
    },

    # Ordered mask list for deterministic output column order.
    # For sessioned datasets, masks are located under:
    #   brain_coverage/<subj>/<ses>/masks/
    # and mask filenames do NOT include the session token.
    "mask_templates": OrderedDict([
        ("icbm152", "{braincov_root}/{subj}/{ses}/masks/{subj}_space-ACPC_mni_icbm152_brain_coverage_mask.nii.gz"),
        ("cerebrum", "{braincov_root}/{subj}/{ses}/masks/{subj}_space-ACPC_mni_cerebrum_brain_coverage_mask.nii.gz"),
        ("upper_cerebrum", "{braincov_root}/{subj}/{ses}/masks/{subj}_space-ACPC_mni_upper_cerebrum_brain_coverage_mask.nii.gz"),
        ("lower_cerebrum", "{braincov_root}/{subj}/{ses}/masks/{subj}_space-ACPC_mni_lower_cerebrum_brain_coverage_mask.nii.gz"),
        ("cerebellum_and_midbrain", "{braincov_root}/{subj}/{ses}/masks/{subj}_space-ACPC_mni_cerebellum_and_midbrain_brain_coverage_mask.nii.gz"),
    ]),

    "options": {
        "allow_wildcard_fallback": True,
        "keep_intermediates": False,
        "verbose": True,
    }
}


def count_nonzero_voxels(img_path: str) -> float:
    data = nib.load(img_path).get_fdata()
    return float((data != 0).sum())


def get_subject_list_for_session(qsiprep_root: str, ses: str):
    """Discover subjects for a given session by scanning qsiprep_root for
    sub-*/<ses>/ directories."""
    subs = sorted(
        os.path.basename(p)
        for p in glob(os.path.join(qsiprep_root, "sub-*"))
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, ses))
    )
    return subs


def find_preproc_dwi(subj: str, ses: str, qsiprep_root: str, dwi_prefix: str, allow_fallback: bool) -> Optional[str]:
    """
    Expected sessioned filename:
      <qsiprep_root>/<subj>/<ses>/dwi/<subj>_<ses>_<dwi_prefix>_desc-preproc_dwi.nii.gz
    Example:
      sub-NDARAF078EUY_ses-1_dir-PA_space-ACPC_desc-preproc_dwi.nii.gz
    """
    expected = os.path.join(
        qsiprep_root, subj, ses, "dwi",
        f"{subj}_{ses}_{dwi_prefix}_desc-preproc_dwi.nii.gz",
    )
    if os.path.exists(expected):
        return expected

    if not allow_fallback:
        return None

    pattern = os.path.join(qsiprep_root, subj, ses, "dwi", f"{subj}_{ses}_*_desc-preproc_dwi.nii.gz")
    candidates = sorted(glob(pattern))
    if candidates:
        print(f"  [WARN] Expected DWI not found for {subj} {ses}.")
        print(f"         Falling back to first match: {os.path.basename(candidates[0])}")
        return candidates[0]

    return None


def compute_coverage(subj: str, ses: str, dwi_file: str, mask_file: str, work_dir: str) -> Optional[float]:
    subj_tmp = os.path.join(work_dir, ses, subj)
    os.makedirs(subj_tmp, exist_ok=True)

    dwi_float = os.path.join(subj_tmp, f"{subj}_{ses}_dwi_float.nii.gz")
    dwi_mean = os.path.join(subj_tmp, f"{subj}_{ses}_dwi_meanT.nii.gz")
    dwi_mean_bin = os.path.join(subj_tmp, f"{subj}_{ses}_dwi_meanT_bin.nii.gz")
    masked = os.path.join(subj_tmp, f"{subj}_{ses}_masked.nii.gz")

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


def main():
    base = CONFIG["base_path"]
    dataset = CONFIG["dataset_name"]

    dwi_prefix = os.environ.get("DWI_PREFIX", CONFIG["dwi_prefix"])

    qsiprep_root = CONFIG["paths"]["qsiprep_root"].format(base=base, dataset=dataset)
    braincov_root = CONFIG["paths"]["braincov_root"].format(base=base, dataset=dataset)
    results_dir = CONFIG["paths"]["results_dir"].format(base=base, dataset=dataset)
    os.makedirs(results_dir, exist_ok=True)

    work_dir = os.path.join(results_dir, "tmp")
    os.makedirs(work_dir, exist_ok=True)

    start = datetime.now()
    print(f"Started at {start}")
    print(f"Dataset: {dataset}")
    print(f"QSIPrep root: {qsiprep_root}")
    print(f"BrainCov root: {braincov_root}")
    print(f"DWI prefix (sessioned): {dwi_prefix}")
    print(f"Allow wildcard fallback: {CONFIG['options']['allow_wildcard_fallback']}")

    for ses in CONFIG["sessions"]:
        out_csv = CONFIG["paths"]["output_csv_template"].format(base=base, dataset=dataset, ses=ses)

        # Build mask templates with session in path
        mask_templates = OrderedDict(
            (k, v.format(braincov_root=braincov_root, subj="{subj}", ses=ses))
            for k, v in CONFIG["mask_templates"].items()
        )

        subjects = get_subject_list_for_session(qsiprep_root, ses)
        total = len(subjects)

        print(f"\n--- Processing {ses} ({total} subjects) ---")
        print(f"Output CSV: {out_csv}")

        rows = []
        for i, subj in enumerate(subjects, 1):
            print(f"[{ses} {i}/{total}] Processing {subj} ...", flush=True)

            dwi_file = find_preproc_dwi(
                subj=subj,
                ses=ses,
                qsiprep_root=qsiprep_root,
                dwi_prefix=dwi_prefix,
                allow_fallback=CONFIG["options"]["allow_wildcard_fallback"],
            )

            row = {"participant_id": subj}

            if dwi_file is None:
                for name in mask_templates:
                    row[f"coverage_{name}"] = None
                rows.append(row)
                continue

            for name, tmpl in mask_templates.items():
                mask_file = tmpl.format(subj=subj)

                if not os.path.exists(mask_file):
                    if CONFIG["options"]["verbose"]:
                        print(f"  [WARN] Missing mask for {subj} {ses}: {mask_file}")
                    row[f"coverage_{name}"] = None
                    continue

                row[f"coverage_{name}"] = compute_coverage(subj, ses, dwi_file, mask_file, work_dir)

            rows.append(row)

            if not CONFIG["options"]["keep_intermediates"]:
                shutil.rmtree(os.path.join(work_dir, ses, subj), ignore_errors=True)

        df = pd.DataFrame(rows)

        # Enforce deterministic column order in output CSV
        ordered_cols = ["participant_id"] + [f"coverage_{k}" for k in mask_templates.keys()]
        df = df.reindex(columns=ordered_cols)

        df.to_csv(out_csv, index=False)
        print(f"Saved {ses} brain coverage results to: {out_csv}")

    print(f"\nTotal runtime: {datetime.now() - start}")

    if not CONFIG["options"]["keep_intermediates"]:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
