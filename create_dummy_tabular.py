#!/usr/bin/env python3
"""
create_dummy_tabular.py
Creates the dummy tabular/pickle files that WholeHeartRL's data pipeline expects.

The original code was designed for UK Biobank which has extensive tabular data
(health conditions, biomarkers, etc). Since we're using ACDC, we create
minimal dummy versions of these files so the data pipeline doesn't crash.

Usage:
    python create_dummy_tabular.py --data_dir /path/to/data/raw --output_dir /path/to/data
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to converted raw data (data/raw)")
    parser.add_argument("--output_dir", required=True, help="Path to data/ root")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    # Find all patient IDs (folder names in data/raw/)
    patient_ids = sorted([
        int(d) for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()
    ])

    print(f"Found {len(patient_ids)} patients: {patient_ids[:5]}...{patient_ids[-5:]}")

    # Create directories
    tabular_dir = os.path.join(output_dir, "tabular")
    dataloader_dir = os.path.join(output_dir, "dataloader")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(tabular_dir, exist_ok=True)
    os.makedirs(dataloader_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Create dummy "all features" tabular CSV
    # WholeHeartRL uses this to filter healthy subjects
    # We create a minimal CSV with required columns
    n = len(patient_ids)
    df_all = pd.DataFrame({
        "eid": patient_ids,
    })
    # Add dummy columns for health conditions (indices 977-981)
    for i in range(977, 982):
        df_all[f"col_{i}"] = 1  # "Excellent" health
    # Add NaN columns for disease dates (so all subjects pass healthy filter)
    df_all["col_10362"] = np.nan  # No obesity
    df_all["col_10229"] = np.nan  # No MI
    df_all["col_10377"] = np.nan  # No acute MI

    all_features_path = os.path.join(tabular_dir, "all_features.csv")
    df_all.to_csv(all_features_path, index=False)
    print(f"Created: {all_features_path}")

    # 2. Create dummy biomarker table
    # WholeHeartRL expects cardiac biomarkers (LVEDV, LVESV, LVEF, etc.)
    biomarker_cols = [
        "eid", "Age",
        "LVEDV (mL)", "LVESV (mL)", "LVSV (mL)", "LVEF (%)",
        "LVCO (L/min)", "LVM (g)",
        "RVEDV (mL)", "RVESV (mL)", "RVSV (mL)", "RVEF (%)",
        "LAV max (mL)", "LAV min (mL)", "LASV (mL)", "LAEF (%)",
        "RAV max (mL)", "RAV min (mL)", "RASV (mL)", "RAEF (%)"
    ]
    rng = np.random.RandomState(42)
    df_bio = pd.DataFrame({
        "eid": patient_ids,
        "Age": rng.randint(40, 70, n),
        "LVEDV (mL)": rng.normal(150, 30, n),
        "LVESV (mL)": rng.normal(60, 15, n),
        "LVSV (mL)": rng.normal(90, 20, n),
        "LVEF (%)": rng.normal(60, 8, n),
        "LVCO (L/min)": rng.normal(5, 1, n),
        "LVM (g)": rng.normal(120, 25, n),
        "RVEDV (mL)": rng.normal(140, 30, n),
        "RVESV (mL)": rng.normal(55, 15, n),
        "RVSV (mL)": rng.normal(85, 20, n),
        "RVEF (%)": rng.normal(55, 8, n),
        "LAV max (mL)": rng.normal(80, 20, n),
        "LAV min (mL)": rng.normal(40, 15, n),
        "LASV (mL)": rng.normal(40, 10, n),
        "LAEF (%)": rng.normal(50, 10, n),
        "RAV max (mL)": rng.normal(75, 20, n),
        "RAV min (mL)": rng.normal(35, 15, n),
        "RASV (mL)": rng.normal(40, 10, n),
        "RAEF (%)": rng.normal(50, 10, n),
    })

    biomarker_path = os.path.join(tabular_dir, "biomarkers.csv")
    df_bio.to_csv(biomarker_path, index=False)
    print(f"Created: {biomarker_path}")

    # 3. Create pickle files that the dataloader expects
    # CMR paths pickle: maps patient IDs to their data paths
    cmr_paths = {}
    for pid in patient_ids:
        cmr_paths[pid] = os.path.join(data_dir, f"{pid:03d}")

    cmr_pickle_path = os.path.join(dataloader_dir, "cmr_subject_paths.pkl")
    with open(cmr_pickle_path, "wb") as f:
        pickle.dump(cmr_paths, f)
    print(f"Created: {cmr_pickle_path}")

    # Biomarker table pickle
    bio_pickle_path = os.path.join(dataloader_dir, "biomarker_table.pkl")
    with open(bio_pickle_path, "wb") as f:
        pickle.dump(df_bio, f)
    print(f"Created: {bio_pickle_path}")

    # Processed table pickle (empty initially)
    proc_pickle_path = os.path.join(dataloader_dir, "processed_table.pkl")
    with open(proc_pickle_path, "wb") as f:
        pickle.dump(pd.DataFrame(), f)
    print(f"Created: {proc_pickle_path}")

    print()
    print("=" * 60)
    print("All tabular/pickle files created successfully.")
    print(f"Tabular dir: {tabular_dir}")
    print(f"Dataloader dir: {dataloader_dir}")
    print(f"Processed dir: {processed_dir}")


if __name__ == "__main__":
    main()
