#!/usr/bin/env python3
"""
prepare_pipeline_data.py

Preprocesses converted ACDC data into the .npz format WholeHeartRL expects,
and creates properly formatted pickle files for the data pipeline.

This bypasses WholeHeartRL's UK Biobank-specific data loading entirely.

Run AFTER convert_acdc.py has created data/raw/001/, data/raw/002/, etc.

Usage:
    python prepare_pipeline_data.py --data_dir ./data/raw --output_dir ./data
"""

import os
import sys
import argparse
import pickle
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path


def process_single_subject(subject_dir, processed_dir, subject_id, image_size=(128, 128), sax_slice_num=6):
    """Process one subject's NIfTI files into a single .npz file."""

    sa_path = os.path.join(subject_dir, "sa.nii.gz")
    seg_path = os.path.join(subject_dir, "seg_sa.nii.gz")
    la_2ch_path = os.path.join(subject_dir, "la_2ch.nii.gz")
    la_3ch_path = os.path.join(subject_dir, "la_3ch.nii.gz")
    la_4ch_path = os.path.join(subject_dir, "la_4ch.nii.gz")

    if not os.path.exists(sa_path):
        return None

    # Load short-axis
    sax = nib.load(sa_path).get_fdata().astype(np.float32)  # (H, W, S, T)
    seg_sax = nib.load(seg_path).get_fdata().astype(np.float32)

    H, W, S, T = sax.shape
    # Standardize time dimension to 20 frames
    target_T = 20
    if T > target_T:
        # Subsample evenly
        indices = np.linspace(0, T - 1, target_T, dtype=int)
        sax = sax[:, :, :, indices]
        seg_sax = seg_sax[:, :, :, indices]
        T = target_T
    elif T < target_T:
        # Pad with zeros
        pad_t = target_T - T
        sax = np.pad(sax, [(0,0), (0,0), (0,0), (0, pad_t)], mode='constant')
        seg_sax = np.pad(seg_sax, [(0,0), (0,0), (0,0), (0, pad_t)], mode='constant')
        T = target_T
    # Center crop or pad spatially to image_size
    def crop_or_pad_2d(arr, target_h, target_w):
        """Center crop or pad to target size. arr shape: (H, W, ...)"""
        h, w = arr.shape[0], arr.shape[1]
        # Pad if needed
        pad_h = max(target_h - h, 0)
        pad_w = max(target_w - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bot = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_widths = [(pad_top, pad_bot), (pad_left, pad_right)] + [(0, 0)] * (len(arr.shape) - 2)
            arr = np.pad(arr, pad_widths, mode='constant', constant_values=0)
        # Crop
        h, w = arr.shape[0], arr.shape[1]
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return arr[start_h:start_h + target_h, start_w:start_w + target_w]

    sax = crop_or_pad_2d(sax, image_size[0], image_size[1])
    seg_sax = crop_or_pad_2d(seg_sax, image_size[0], image_size[1])

    # Handle slice dimension: select center slices if more than sax_slice_num
    S = sax.shape[2]
    if S > sax_slice_num:
        start = (S - sax_slice_num) // 2
        sax = sax[:, :, start:start + sax_slice_num, :]
        seg_sax = seg_sax[:, :, start:start + sax_slice_num, :]
    elif S < sax_slice_num:
        # Pad with zeros along slice dimension
        pad_s = sax_slice_num - S
        sax = np.pad(sax, [(0, 0), (0, 0), (0, pad_s), (0, 0)], mode='constant')
        seg_sax = np.pad(seg_sax, [(0, 0), (0, 0), (0, pad_s), (0, 0)], mode='constant')

    # Load long-axis (these are dummy zeros for ACDC)
    la_views = []
    for la_path in [la_2ch_path, la_3ch_path, la_4ch_path]:
        if os.path.exists(la_path):
            la = nib.load(la_path).get_fdata().astype(np.float32)
            la = crop_or_pad_2d(la, image_size[0], image_size[1])
        else:
            la = np.zeros((image_size[0], image_size[1], 1, T), dtype=np.float32)
        la_views.append(la)

    # Stack LA views: (H, W, 3, T)
    lax = np.concatenate(la_views, axis=2)
    seg_lax = np.zeros_like(lax, dtype=np.float32)

    # Ensure LAX has correct time dimension
    if lax.shape[-1] != T:
        # Pad or crop time dimension to match SAX
        if lax.shape[-1] < T:
            pad_t = T - lax.shape[-1]
            lax = np.pad(lax, [(0, 0), (0, 0), (0, 0), (0, pad_t)], mode='constant')
            seg_lax = np.pad(seg_lax, [(0, 0), (0, 0), (0, 0), (0, pad_t)], mode='constant')
        else:
            lax = lax[:, :, :, :T]
            seg_lax = seg_lax[:, :, :, :T]

    # Save as .npz
    out_dir = os.path.join(processed_dir, subject_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "processed_seg_allax.npz")

    np.savez(out_path,
             sax=sax,           # (128, 128, 6, T)
             lax=lax,           # (128, 128, 3, T)
             seg_sax=seg_sax,   # (128, 128, 6, T)
             seg_lax=seg_lax)   # (128, 128, 3, T)

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to data/raw")
    parser.add_argument("--output_dir", required=True, help="Path to data/")
    parser.add_argument("--num_train", type=int, default=70)
    parser.add_argument("--num_val", type=int, default=15)
    parser.add_argument("--num_test", type=int, default=15)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    processed_dir = os.path.join(output_dir, "processed")
    dataloader_dir = os.path.join(output_dir, "dataloader")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(dataloader_dir, exist_ok=True)

    # Find all subjects
    subjects = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    total_needed = args.num_train + args.num_val + args.num_test
    print(f"Found {len(subjects)} subjects. Need {total_needed} (train={args.num_train}, val={args.num_val}, test={args.num_test})")

    if len(subjects) < total_needed:
        print(f"WARNING: Only {len(subjects)} subjects available, adjusting splits...")
        args.num_train = int(len(subjects) * 0.7)
        args.num_val = int(len(subjects) * 0.15)
        args.num_test = len(subjects) - args.num_train - args.num_val
        total_needed = args.num_train + args.num_val + args.num_test
        print(f"Adjusted: train={args.num_train}, val={args.num_val}, test={args.num_test}")

    # Process all subjects
    print("\nProcessing subjects into .npz format...")
    processed_paths = []
    processed_ids = []
    for i, subject_id in enumerate(subjects[:total_needed]):
        subject_dir = os.path.join(data_dir, subject_id)
        out_path = process_single_subject(subject_dir, processed_dir, subject_id)
        if out_path is not None:
            processed_paths.append(str(out_path))
            processed_ids.append(int(subject_id))
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{total_needed}")
        else:
            print(f"  SKIP: {subject_id}")

    print(f"\nSuccessfully processed: {len(processed_paths)}")

    # Verify we have enough
    if len(processed_paths) < total_needed:
        args.num_train = int(len(processed_paths) * 0.7)
        args.num_val = int(len(processed_paths) * 0.15)
        args.num_test = len(processed_paths) - args.num_train - args.num_val
        total_needed = len(processed_paths)

    # Split into train/val/test
    np.random.seed(42)
    indices = np.random.permutation(len(processed_paths))

    train_idx = indices[:args.num_train]
    val_idx = indices[args.num_train:args.num_train + args.num_val]
    test_idx = indices[args.num_train + args.num_val:total_needed]

    train_paths = [processed_paths[i] for i in train_idx]
    val_paths = [processed_paths[i] for i in val_idx]
    test_paths = [processed_paths[i] for i in test_idx]

    print(f"\nSplit: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    # Save cmr_subject_paths.pkl (the paths dict the dataloader expects)
    paths_dict = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
    }
    cmr_pickle_path = os.path.join(dataloader_dir, "cmr_subject_paths.pkl")
    with open(cmr_pickle_path, "wb") as f:
        pickle.dump(paths_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {cmr_pickle_path}")

    # Create biomarker table with eid_87802 column (what the code expects)
    rng = np.random.RandomState(42)
    all_ids = [processed_ids[i] for i in indices[:total_needed]]

    df_bio = pd.DataFrame({
        "eid_87802": all_ids,
        "Age": rng.randint(40, 70, len(all_ids)),
        "LVEDV (mL)": rng.normal(150, 30, len(all_ids)),
        "LVESV (mL)": rng.normal(60, 15, len(all_ids)),
        "LVSV (mL)": rng.normal(90, 20, len(all_ids)),
        "LVEF (%)": rng.normal(60, 8, len(all_ids)),
        "LVCO (L/min)": rng.normal(5, 1, len(all_ids)),
        "LVM (g)": rng.normal(120, 25, len(all_ids)),
        "RVEDV (mL)": rng.normal(140, 30, len(all_ids)),
        "RVESV (mL)": rng.normal(55, 15, len(all_ids)),
        "RVSV (mL)": rng.normal(85, 20, len(all_ids)),
        "RVEF (%)": rng.normal(55, 8, len(all_ids)),
        "LAV max (mL)": rng.normal(80, 20, len(all_ids)),
        "LAV min (mL)": rng.normal(40, 15, len(all_ids)),
        "LASV (mL)": rng.normal(40, 10, len(all_ids)),
        "LAEF (%)": rng.normal(50, 10, len(all_ids)),
        "RAV max (mL)": rng.normal(75, 20, len(all_ids)),
        "RAV min (mL)": rng.normal(35, 15, len(all_ids)),
        "RASV (mL)": rng.normal(40, 10, len(all_ids)),
        "RAEF (%)": rng.normal(50, 10, len(all_ids)),
    })

    bio_pickle_path = os.path.join(dataloader_dir, "biomarker_table.pkl")
    df_bio.to_pickle(bio_pickle_path)
    print(f"Saved: {bio_pickle_path}")

    # Verify a processed file
    sample = np.load(str(processed_paths[0]))
    print(f"\nVerification — sample .npz contents:")
    for key in sample.files:
        print(f"  {key}: shape={sample[key].shape}, dtype={sample[key].dtype}")

    print(f"\n{'='*60}")
    print("Done! Now re-upload to Modal:")
    print("  modal volume put cardiac-data data/processed /processed")
    print("  modal volume put cardiac-data data/dataloader /dataloader")


if __name__ == "__main__":
    main()
