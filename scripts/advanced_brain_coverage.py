#!/usr/bin/env python3
# =============================================================================
# Purpose: Calculate brain coverage for each subject’s DWI data using a
#          subject-specific ACPC-space brain mask and output results to QC dir.
#
# Adapted from: https://github.com/DCAN-Labs/brain_coverage
# Created on 11/13/25 by Samantha Keppler
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Dataset-level identifiers
    "base_path": "/mnt/synapse/neurocat-lab/R21MH133229_asd_dmri_lifespan/datasets_v1.0",
    "dataset_name": "abideii-ip",
    "qsiprep_version": "qsiprep-1.0.0rc2",

    # Diffusion metadata
    "dir_tag": "PA",

    # Directory structure
    "dirs": {
        "derivatives": "derivatives",
        "qc": "qc",
        "tmp": "tmp",
        "dwi": "dwi",
    },

    # Path templates
    "paths": {
        "bids_root": "{base}/{dataset}",

        "qsiprep_root": (
            "{base}/{dataset}/{derivatives}/{qsiprep}"
        ),

        "subject_root": (
            "{base}/{dataset}/{derivatives}/{qsiprep}/{subj}"
        ),

        "brain_mask": (
            "{base}/{dataset}/{derivatives}/{qsiprep}/{subj}/dwi/"
            "{subj}_space-ACPC_mni_icbm152_brain_coverage_mask.nii.gz"
        ),

        "dwi_preproc": (
            "{base}/{dataset}/{derivatives}/{qsiprep}/{subj}/dwi/"
            "{subj}_dir-*_space-ACPC_desc-preproc_dwi.nii.gz"
        ),

        "qc_dir": (
            "{base}/{dataset}/{derivatives}/{qsiprep}/{qc}"
        ),

        "output_tsv": (
            "{base}/{dataset}/{derivatives}/{qsiprep}/{qc}/brain_coverage.tsv"
        ),
    },

    # Runtime options
    "options": {
        "keep_intermediates": False,
        "verbose": True,
    },
}

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import shutil
from glob import glob
from datetime import datetime

import pandas as pd
import nibabel as nib
from nibabel import imagestats
from nipype.interfaces import fsl
from nipype.interfaces.fsl.maths import MathsCommand

# =============================================================================
# HELPERS
# =============================================================================

class LazyDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def get_cli_args_from_config(config):
    base = config["base_path"]
    dataset = config["dataset_name"]
    qsiprep = config["qsiprep_version"]
    dirs = config["dirs"]

    paths = {
        k: v.format(
            base=base,
            dataset=dataset,
            qsiprep=qsiprep,
            derivatives=dirs["derivatives"],
            qc=dirs["qc"],
            subj="{subj}",
        )
        for k, v in config["paths"].items()
    }

    qc_dir = paths["qc_dir"]

    return {
        "paths": paths,
        "qc_dir": qc_dir,
        "work_dir": os.path.join(qc_dir, dirs["tmp"]),
        "failed_file": os.path.join(qc_dir, "failed_bc_runs.txt"),
        "keep": config["options"]["keep_intermediates"],
        "verbose": config["options"]["verbose"],
    }


def get_subject_list(qsiprep_root):
    subj_paths = [
        p for p in glob(os.path.join(qsiprep_root, "sub-*"))
        if os.path.isdir(p)
    ]
    subj_ids = [os.path.basename(p) for p in subj_paths]

    if not subj_ids:
        sys.exit(f"No subject folders found under {qsiprep_root}")

    return subj_ids


def get_image_paths(subj, cli_args):
    temp_dir = os.path.join(cli_args["work_dir"], subj)
    os.makedirs(temp_dir, exist_ok=True)

    path_to = LazyDict({
        "MNI": cli_args["paths"]["brain_mask"].format(subj=subj),
        "temp_dir": temp_dir,
        "prefiltered": os.path.join(temp_dir, f"{subj}_prefiltered.nii.gz"),
        "bold": os.path.join(temp_dir, f"{subj}_bold_mean.nii.gz"),
        "mask": os.path.join(temp_dir, f"{subj}_bold_mask.nii.gz"),
        "mask_masked": os.path.join(temp_dir, f"{subj}_bold_mask_masked.nii.gz"),
    })

    return path_to


def calculate_coverage(subj, cli_args):
    path_to = get_image_paths(subj, cli_args)

    dwi_files = glob(
        cli_args["paths"]["dwi_preproc"].format(subj=subj)
    )

    if not dwi_files:
        print(f"No DWI file found for {subj}")
        return None

    dwi_file = dwi_files[0]

    MathsCommand(
        in_file=dwi_file,
        out_file=path_to.prefiltered,
        output_datatype="float",
        output_type="NIFTI_GZ",
    ).run()

    if not os.path.exists(path_to.prefiltered):
        print(f"Failed to create prefiltered file for {subj}")
        return None

    fsl.MeanImage(
        in_file=path_to.prefiltered,
        dimension="T",
        out_file=path_to.bold,
        output_type="NIFTI_GZ",
    ).run()

    fsl.UnaryMaths(
        in_file=path_to.bold,
        operation="bin",
        out_file=path_to.mask,
        output_type="NIFTI_GZ",
    ).run()

    fsl.ApplyMask(
        in_file=path_to.mask,
        mask_file=path_to.MNI,
        out_file=path_to.mask_masked,
        output_type="NIFTI_GZ",
    ).run()

    n_vox = {
        img: float(imagestats.count_nonzero_voxels(nib.load(path_to[img])))
        for img in ("mask_masked", "MNI")
    }

    coverage = round((n_vox["mask_masked"] / n_vox["MNI"]) * 100, 3)

    if not cli_args["keep"]:
        shutil.rmtree(path_to.temp_dir, ignore_errors=True)

    return coverage


# =============================================================================
# MAIN
# =============================================================================

def main():
    cli_args = get_cli_args_from_config(CONFIG)

    os.makedirs(cli_args["qc_dir"], exist_ok=True)
    os.makedirs(cli_args["work_dir"], exist_ok=True)

    start = datetime.now()
    print(f"Started at {start}")

    subjects = get_subject_list(
        cli_args["paths"]["qsiprep_root"]
    )

    results = []
    total = len(subjects)

    for i, subj in enumerate(subjects, 1):
        print(f"[{i}/{total}] Calculating coverage for {subj} ...", flush=True)
        cov = calculate_coverage(subj, cli_args)
        if cov is not None:
            results.append({
                "participant_id": subj,
                "brain_coverage": cov,
            })

    if not results:
        sys.exit("No coverage values calculated. Check DWI paths.")

    df = pd.DataFrame(results)
    df.to_csv(cli_args["paths"]["output_tsv"], sep="\t", index=False)

    print(f"\nSaved QC results to: {cli_args['paths']['output_tsv']}")
    print(f"Total runtime: {datetime.now() - start}")


if __name__ == "__main__":
    main()


