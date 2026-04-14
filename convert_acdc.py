#!/usr/bin/env python3
"""
convert_acdc.py
Converts ACDC dataset to WholeHeartRL expected format.

ACDC structure (input):
    ACDC/database/training/patient001/
        patient001_4d.nii.gz          # Full 4D cine (H, W, S, T)
        patient001_frame01.nii.gz     # ED frame
        patient001_frame01_gt.nii.gz  # ED segmentation
        patient001_frame12.nii.gz     # ES frame  
        patient001_frame12_gt.nii.gz  # ES segmentation
        Info.cfg

WholeHeartRL structure (output):
    data/raw/<patient_id>/
        sa.nii.gz        # Short-axis 4D cine (H, W, S, T)
        seg_sa.nii.gz    # Segmentation (H, W, S, T) - only ED/ES frames labeled
        la_2ch.nii.gz    # Dummy long-axis (zeros)
        la_3ch.nii.gz    # Dummy long-axis (zeros)
        la_4ch.nii.gz    # Dummy long-axis (zeros)

Usage:
    python convert_acdc.py --acdc_dir /path/to/ACDC/database/training --output_dir /path/to/data/raw
"""

import os
import sys
import glob
import shutil
import argparse
import numpy as np
import nibabel as nib


def parse_info_cfg(info_path):
    """Parse ACDC Info.cfg to get ED and ES frame indices."""
    info = {}
    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                info[key.strip()] = val.strip()
    ed_frame = int(info.get("ED", 0))
    es_frame = int(info.get("ES", 0))
    group = info.get("Group", "unknown")
    return ed_frame, es_frame, group


def create_full_segmentation(nii_4d, ed_seg_path, es_seg_path, ed_frame, es_frame):
    """
    Create a full 4D segmentation volume.
    Only ED and ES frames have labels; all other frames are zeros.
    """
    shape_4d = nii_4d.shape  # (H, W, S, T)
    seg_4d = np.zeros(shape_4d, dtype=np.int16)

    if os.path.exists(ed_seg_path):
        ed_seg = nib.load(ed_seg_path).get_fdata().astype(np.int16)
        seg_4d[:, :, :, ed_frame] = ed_seg[:, :, :]

    if os.path.exists(es_seg_path):
        es_seg = nib.load(es_seg_path).get_fdata().astype(np.int16)
        seg_4d[:, :, :, es_frame] = es_seg[:, :, :]

    return seg_4d


def create_dummy_lax(shape_hw, num_timeframes, affine):
    """
    Create a dummy long-axis NIfTI with zeros.
    Shape: (H, W, 1, T) — single slice, all timeframes.
    """
    h, w = shape_hw
    dummy = np.zeros((h, w, 1, num_timeframes), dtype=np.float32)
    return nib.Nifti1Image(dummy, affine)


def convert_patient(patient_dir, output_dir, patient_id):
    """Convert a single ACDC patient to WholeHeartRL format."""

    # Find files
    nii_4d_path = os.path.join(patient_dir, f"{patient_id}_4d.nii.gz")
    info_path = os.path.join(patient_dir, "Info.cfg")

    if not os.path.exists(nii_4d_path):
        print(f"  SKIP {patient_id}: no 4D file found")
        return False

    if not os.path.exists(info_path):
        print(f"  SKIP {patient_id}: no Info.cfg found")
        return False

    # Parse info
    ed_frame, es_frame, group = parse_info_cfg(info_path)

    # Find segmentation files
    # ACDC naming: patient001_frame01_gt.nii.gz
    ed_frame_str = f"{ed_frame:02d}"
    es_frame_str = f"{es_frame:02d}"
    ed_seg_path = os.path.join(patient_dir, f"{patient_id}_frame{ed_frame_str}_gt.nii.gz")
    es_seg_path = os.path.join(patient_dir, f"{patient_id}_frame{es_frame_str}_gt.nii.gz")

    # Load 4D cine volume
    nii_4d = nib.load(nii_4d_path)
    data_4d = nii_4d.get_fdata()
    affine = nii_4d.affine

    if len(data_4d.shape) != 4:
        print(f"  SKIP {patient_id}: unexpected shape {data_4d.shape}")
        return False

    H, W, S, T = data_4d.shape
    print(f"  {patient_id}: shape=({H},{W},{S},{T}), ED={ed_frame}, ES={es_frame}, group={group}")

    # Create output directory
    # Use numeric ID for WholeHeartRL compatibility (it expects integer folder names)
    numeric_id = patient_id.replace("patient", "")
    out_patient_dir = os.path.join(output_dir, numeric_id)
    os.makedirs(out_patient_dir, exist_ok=True)

    # 1. Save short-axis 4D cine as sa.nii.gz
    sa_nii = nib.Nifti1Image(data_4d.astype(np.float32), affine)
    nib.save(sa_nii, os.path.join(out_patient_dir, "sa.nii.gz"))

    # 2. Create and save full 4D segmentation
    seg_4d = create_full_segmentation(data_4d, ed_seg_path, es_seg_path, ed_frame, es_frame)
    seg_nii = nib.Nifti1Image(seg_4d, affine)
    nib.save(seg_nii, os.path.join(out_patient_dir, "seg_sa.nii.gz"))

    # 3. Create dummy long-axis views (zeros)
    for la_name in ["la_2ch.nii.gz", "la_3ch.nii.gz", "la_4ch.nii.gz"]:
        la_nii = create_dummy_lax((H, W), T, affine)
        nib.save(la_nii, os.path.join(out_patient_dir, la_name))

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert ACDC to WholeHeartRL format")
    parser.add_argument("--acdc_dir", required=True, help="Path to ACDC/database/training")
    parser.add_argument("--output_dir", required=True, help="Path to output data/raw directory")
    args = parser.parse_args()

    acdc_dir = args.acdc_dir
    output_dir = args.output_dir

    if not os.path.exists(acdc_dir):
        print(f"ERROR: ACDC directory not found: {acdc_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Find all patient directories
    patient_dirs = sorted(glob.glob(os.path.join(acdc_dir, "patient*")))
    print(f"Found {len(patient_dirs)} patients in {acdc_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    success = 0
    failed = 0
    for patient_dir in patient_dirs:
        patient_id = os.path.basename(patient_dir)
        ok = convert_patient(patient_dir, output_dir, patient_id)
        if ok:
            success += 1
        else:
            failed += 1

    print("=" * 60)
    print(f"Done. Converted: {success}, Skipped: {failed}")
    print(f"Output at: {output_dir}")
    print()
    print("Next step: verify with")
    print(f"  ls {output_dir}/001/")
    print("  You should see: sa.nii.gz  seg_sa.nii.gz  la_2ch.nii.gz  la_3ch.nii.gz  la_4ch.nii.gz")


if __name__ == "__main__":
    main()
