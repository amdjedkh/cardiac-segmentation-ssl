"""
modal_seg.py — Segmentation finetuning on ACDC

Runs two sequential jobs on Modal:
  1. Pretrained condition: SSL encoder loaded from MAE checkpoint
  2. Scratch condition:    Random initialisation, no pretrained weights

Usage:
    modal run modal_seg.py                      # Both conditions
    modal run modal_seg.py::main --condition pretrained
    modal run modal_seg.py::main --condition scratch

Checkpoints are written to:
    /vol/logs/checkpoints/seg_pretrained/<timestamp>/
    /vol/logs/checkpoints/seg_scratch/<timestamp>/

Configs are read from /vol/WholeHeartRL/configs/.
Both configs must be uploaded to the Modal volume before running:
    modal volume put cardiac-data \
        configs/config_segmentation_acdc_pretrained.yaml \
        WholeHeartRL/configs/config_segmentation_acdc_pretrained.yaml
    modal volume put cardiac-data \
        configs/config_segmentation_acdc_scratch.yaml \
        WholeHeartRL/configs/config_segmentation_acdc_scratch.yaml
"""

import modal

app = modal.App("wholeheart-seg")
vol = modal.Volume.from_name("cardiac-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
    )
    .pip_install(
        "moviepy",
        "lightning>=2.1",
        "monai>=1.3",
        "nibabel",
        "numpy",
        "pandas",
        "scipy",
        "PyYAML",
        "wandb",
        "timm==0.9.16",
        "matplotlib",
        "tqdm",
        "h5py",
        "Pillow",
        "opencv-python-headless",
        "scikit-learn",
        "lpips",
        "pytorch-ignite",
        "torchsummary",
        "medutils-mri",
    )
)


def _setup_environment():
    """Set all environment variables required by the codebase."""
    import os
    import sys

    os.environ["PYTHONPATH"] = "/vol/WholeHeartRL"
    os.environ["DATA_ROOT"] = "/vol/raw"
    os.environ["ALL_FEATURE_TABULAR_DIR"] = "/vol/tabular/all_features.csv"
    os.environ["BIOMARKER_TABULAR_DIR"] = "/vol/tabular/biomarkers.csv"
    os.environ["LOG_FOLDER"] = "/vol/logs"
    os.environ["PROCESS_ROOT"] = "/vol/processed"
    os.environ["DATALOADER_FILE_ROOT"] = "/vol/dataloader"
    os.environ["CMR_PATH_PICKLE_NAME"] = "cmr_subject_paths.pkl"
    os.environ["BIOMARKER_PICKLE_NAME"] = "biomarker_table.pkl"
    os.environ["PROCESSED_PICKLE_NAME"] = "processed_table.pkl"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    os.makedirs("/vol/logs", exist_ok=True)
    sys.path.insert(0, "/vol/WholeHeartRL")

    # Patch torch.load for PyTorch 2.6 compatibility
    import torch
    _original_torch_load = torch.load
    def _patched_torch_load(f, map_location=None, pickle_module=None, weights_only=False, **kwargs):
        return _original_torch_load(f, map_location=map_location, weights_only=False, **kwargs)
    torch.load = _patched_torch_load
    # Patch Lightning cloud_io for PyTorch 2.6 weights_only compatibility
    filepath = "/usr/local/lib/python3.11/site-packages/lightning/fabric/utilities/cloud_io.py"
    with open(filepath, "r") as f:
        lines = f.readlines()
    new_lines = [l.replace("weights_only=weights_only,", "weights_only=False,") for l in lines]
    with open(filepath, "w") as f:
        f.writelines(new_lines)

def _print_gpu_info():
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 8,   # 8h ceiling — 300 epochs on 70 subjects is fast but safe margin
    volumes={"/vol": vol},
)
def train_segmentation(condition: str):
    """
    Run segmentation finetuning for one condition.

    Args:
        condition: "pretrained" or "scratch"
    """
    import subprocess
    import sys

    assert condition in ("pretrained", "scratch", "pretrained_20", "scratch_20"), \
        f"condition must be pretrained/scratch/pretrained_20/scratch_20, got '{condition}'"

    _setup_environment()

    print(f"\n{'='*60}")
    print(f"  SEGMENTATION FINETUNING — {condition.upper()} CONDITION")
    print(f"{'='*60}\n")
    _print_gpu_info()

    config_name = f"config_segmentation_acdc_{condition}.yaml"
    config_path = f"/vol/WholeHeartRL/configs/{config_name}"

    # Verify config and checkpoint existence before launching training
    import os
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Upload it with:\n"
            f"  modal volume put cardiac-data configs/{config_name} "
            f"WholeHeartRL/configs/{config_name}"
        )

    if condition == "pretrained":
        import glob
        ckpt = "/vol/logs/checkpoints/run1/14-04-2026_09-14-26/model-epoch=034-val_PSNR=19.78.ckpt"
        if not os.path.exists(ckpt):
            candidates = glob.glob("/vol/logs/checkpoints/**/*.ckpt", recursive=True)
            candidates = [c for c in candidates if "val_PSNR" in c and "run2" in c]
            if not candidates:
                raise FileNotFoundError("No pretraining checkpoint found.")
            ckpt = max(candidates, key=lambda x: float(x.split("val_PSNR=")[1].replace(".ckpt", "")))
            print(f"WARNING: Using fallback checkpoint: {ckpt}")
        print(f"Checkpoint: {ckpt}")

    print(f"\nConfig:    {config_path}")
    print(f"Condition: {condition}")
    print("=" * 60)
    print("Starting training...\n")

    result = subprocess.run(
        [
            sys.executable, "/vol/WholeHeartRL/main.py", "train",
            "-c", config_path,
            "-g", f"seg_acdc_{condition}",
            "-n", f"seg_{condition}",
        ],
        cwd="/vol/WholeHeartRL",
        env=os.environ.copy(),
        capture_output=False,
    )

    vol.commit()
    print(f"\n{'='*60}")
    print(f"  FINISHED — {condition.upper()} | exit code: {result.returncode}")
    print(f"{'='*60}\n")
    return result.returncode

@app.function(image=image, gpu=None, volumes={"/vol": vol})
def patch_lightning():
    filepath = "/usr/local/lib/python3.11/site-packages/lightning/fabric/utilities/cloud_io.py"
    with open(filepath, "r") as f:
        lines = f.readlines()
    
    patched = 0
    for i, line in enumerate(lines):
        if "weights_only=weights_only," in line:
            lines[i] = line.replace("weights_only=weights_only,", "weights_only=False,")
            patched += 1
            print(f"Patched line {i}: {lines[i].rstrip()}")
    
    with open(filepath, "w") as f:
        f.writelines(lines)
    print(f"Done. Patched {patched} lines.")


@app.function(image=image, gpu=None, volumes={"/vol": vol})
def create_acdc_pickles():
    import pickle
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import glob

    processed_dir = Path("/vol/processed")
    dataloader_dir = Path("/vol/dataloader")
    try:
        dataloader_dir.mkdir(exist_ok=True)
    except FileExistsError:
        pass

    # Find all valid processed subjects
    npz_files = sorted(glob.glob(str(processed_dir / "*/processed_seg_allax.npz")))
    patient_ids = [int(Path(f).parent.name) for f in npz_files]
    print(f"Found {len(patient_ids)} processed subjects: {patient_ids[:5]}...{patient_ids[-5:]}")

    # Split 70/15/15
    import random
    random.seed(1)
    shuffled = patient_ids.copy()
    random.shuffle(shuffled)
    train_ids = shuffled[:85]
    val_ids   = shuffled[85:93]
    test_ids  = shuffled[93:100]

    # Build path lists — each entry is Path to the npz file
    # This is what CMRDataModule expects in paths["train"] etc.
    def make_paths(ids):
        return [processed_dir / f"{i:03d}" / "processed_seg_allax.npz" for i in ids]

    paths = {
        "train": make_paths(train_ids),
        "val":   make_paths(val_ids),
        "test":  make_paths(test_ids),
    }

    with open(dataloader_dir / "cmr_subject_paths.pkl", "wb") as f:
        pickle.dump(paths, f)
    print(f"Saved cmr_subject_paths.pkl: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # Build minimal target_table with eid_87802 column (required by dataset.__getitem__)
    # Values are dummy — only used for regression, not segmentation
    rng = np.random.RandomState(1)
    n = len(patient_ids)
    df = pd.DataFrame({
        "eid_87802": patient_ids,
        "LVM (g)": rng.normal(120, 25, n),
    })
    df.to_pickle(dataloader_dir / "biomarker_table.pkl")
    print(f"Saved biomarker_table.pkl with {n} rows")

    vol.commit()
    print("Done.")

@app.function(image=image, gpu=None, volumes={"/vol": vol})
def fix_dataloader_dir():
    import os, shutil
    # Remove the corrupted dataloader file/dir
    path = "/vol/dataloader"
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
            print("Removed file at /vol/dataloader")
        elif os.path.isdir(path):
            print("/vol/dataloader is already a directory")
    os.makedirs(path, exist_ok=True)
    print("Created /vol/dataloader directory")
    vol.commit()

@app.function(image=image, gpu=None, volumes={"/vol": vol})
def check_pickle_paths():
    import pickle
    with open("/vol/dataloader/cmr_subject_paths.pkl", "rb") as f:
        paths = pickle.load(f)
    print(f"Keys: {list(paths.keys())}")
    print(f"Train paths sample: {paths['train'][:3]}")
    print(f"Total: train={len(paths['train'])}, val={len(paths['val'])}, test={len(paths['test'])}")
    # Check if files actually exist
    import os
    for p in paths['train'][:3]:
        print(f"  exists={os.path.exists(str(p))}: {p}")

@app.function(image=image, gpu=None, volumes={"/vol": vol})
def verify_npz():
    import numpy as np, glob
    files = sorted(glob.glob("/vol/processed/**/*.npz", recursive=True))
    bad = []
    for f in files:
        d = np.load(f)
        shape = d["sax"].shape
        if shape != (128, 128, 6, 2):
            bad.append((f, shape))
    print(f"Total files: {len(files)}")
    print(f"Bad files: {len(bad)}")
    for f, s in bad[:10]:
        print(f"  {f}: {s}")


@app.function(image=image, gpu=None, timeout=3600, volumes={"/vol": vol})
def rebuild_npz_2frame():
    import nibabel as nib
    import numpy as np
    import glob, os

    TARGET_H, TARGET_W, SAX_N = 128, 128, 6

    def center_crop(arr, th, tw):
        h, w = arr.shape[0], arr.shape[1]
        hs = max(0, (h - th) // 2)
        ws = max(0, (w - tw) // 2)
        out = arr[hs:hs+th, ws:ws+tw]
        ph = max(0, th - out.shape[0])
        pw = max(0, tw - out.shape[1])
        if ph or pw:
            pad = [(0,ph),(0,pw)] + [(0,0)]*(arr.ndim-2)
            out = np.pad(out, pad)
        return out

    def select_slices(im, seg, n):
        S = im.shape[2]
        if S <= n:
            pad = n - S
            im  = np.concatenate([im,  np.repeat(im[:,:,-1:,:],  pad, axis=2)], axis=2)
            seg = np.concatenate([seg, np.repeat(seg[:,:,-1:,:], pad, axis=2)], axis=2)
            return im, seg
        lv = (seg == 1).any(axis=(0,1,3))
        z0 = int(lv.argmax()) if lv.any() else (S-n)//2
        z0 = max(0, min(z0-1, S-n))
        return im[:,:,z0:z0+n,:], seg[:,:,z0:z0+n,:]

    def remap(seg):
        out = np.zeros_like(seg)
        out[seg == 1] = 3
        out[seg == 2] = 2
        out[seg == 3] = 1
        return out

    def parse_info(info_path):
        info = {}
        with open(info_path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    info[k.strip()] = v.strip()
        return int(info.get("ED", 0)), int(info.get("ES", 0))

    patient_dirs = sorted(glob.glob("/vol/raw/*/"))
    print(f"Found {len(patient_dirs)} patients")

    for pd_path in patient_dirs:
        pid = os.path.basename(pd_path.rstrip("/"))
        sa_path  = os.path.join(pd_path, "sa.nii.gz")
        seg_path = os.path.join(pd_path, "seg_sa.nii.gz")
        info_path = os.path.join(pd_path, "Info.cfg")

        if not os.path.exists(sa_path):
            print(f"  SKIP {pid}: missing sa.nii.gz")
            continue

        sa  = nib.load(sa_path).get_fdata().astype(np.float32)   # (H,W,S,T)
        seg = nib.load(seg_path).get_fdata().astype(np.int32)
        seg = remap(seg)

        # Extract only ED and ES frames
        if os.path.exists(info_path):
            ed, es = parse_info(info_path)
        else:
            # fallback: find labeled frames
            labeled = np.where(seg.any(axis=(0,1,2)))[0]
            ed, es = (labeled[0], labeled[-1]) if len(labeled) >= 2 else (0, 1)

        sa_2  = sa[..., [ed, es]]    # (H,W,S,2)
        seg_2 = seg[..., [ed, es]]   # (H,W,S,2)

        sa_2  = center_crop(sa_2,  TARGET_H, TARGET_W)
        seg_2 = center_crop(seg_2, TARGET_H, TARGET_W)
        sa_2, seg_2 = select_slices(sa_2, seg_2, SAX_N)

        lax     = np.zeros((TARGET_H, TARGET_W, 3, 2), dtype=np.float32)
        seg_lax = np.zeros((TARGET_H, TARGET_W, 3, 2), dtype=np.float32)

        out_dir = f"/vol/processed/{pid}"
        os.makedirs(out_dir, exist_ok=True)
        np.savez(f"{out_dir}/processed_seg_allax.npz",
                 sax=sa_2.astype(np.float32),
                 lax=lax,
                 seg_sax=seg_2.astype(np.float32),
                 seg_lax=seg_lax)
        print(f"  {pid}: sa={sa_2.shape} seg_unique={np.unique(seg_2)}")

    vol.commit()
    print("Done.")

@app.function(image=image, volumes={"/vol": vol})
def inspect_labels():
    import numpy as np, glob
    files = sorted(glob.glob("/vol/processed/**/*.npz", recursive=True))[:3]
    for f in files:
        d = np.load(f, allow_pickle=True)
        print(f"\n{f}")
        for k in ["seg_sax", "seg_lax"]:
            if k in d:
                arr = d[k]
                print(f"  {k}: shape={arr.shape}, unique={np.unique(arr)}, dtype={arr.dtype}")


@app.function(image=image, gpu=None, timeout=3600*2, volumes={"/vol": vol})
def reprocess_acdc():
    import subprocess, sys
    _setup_environment()
    result = subprocess.run([
        sys.executable, "/vol/convert_acdc.py",
        "--acdc_dir", "/vol/raw",
        "--output_dir", "/vol/processed",
    ], capture_output=False)
    vol.commit()
    return result.returncode


@app.function(image=image, gpu=None, volumes={"/vol": vol})
def inspect_raw():
    import nibabel as nib
    import numpy as np
    for pid in ["001", "002"]:
        seg = nib.load(f"/vol/raw/{pid}/seg_sa.nii.gz").get_fdata()
        sa  = nib.load(f"/vol/raw/{pid}/sa.nii.gz").get_fdata()
        print(f"\n{pid}:")
        print(f"  sa shape:   {sa.shape}")
        print(f"  seg shape:  {seg.shape}")
        print(f"  seg unique: {np.unique(seg)}")

@app.function(image=image, gpu=None, timeout=3600*2, volumes={"/vol": vol})
def rebuild_npz():
    import nibabel as nib
    import numpy as np
    import glob, os

    TARGET_H, TARGET_W, TARGET_T, SAX_N = 128, 128, 50, 6

    def center_crop(arr, th, tw):
        h, w = arr.shape[0], arr.shape[1]
        hs = max(0, (h - th) // 2)
        ws = max(0, (w - tw) // 2)
        out = arr[hs:hs+th, ws:ws+tw]
        ph = max(0, th - out.shape[0])
        pw = max(0, tw - out.shape[1])
        if ph or pw:
            pad = [(0,ph),(0,pw)] + [(0,0)]*(arr.ndim-2)
            out = np.pad(out, pad)
        return out

    def pad_time(arr, tt):
        T = arr.shape[-1]
        if T >= tt: return arr[..., :tt]
        return np.concatenate([arr, np.repeat(arr[...,-1:], tt-T, axis=-1)], axis=-1)

    def select_slices(im, seg, n):
        S = im.shape[2]
        if S <= n:
            pad = n - S
            im  = np.concatenate([im,  np.repeat(im[:,:,-1:,:],  pad, axis=2)], axis=2)
            seg = np.concatenate([seg, np.repeat(seg[:,:,-1:,:], pad, axis=2)], axis=2)
            return im, seg
        lv = (seg == 1).any(axis=(0,1,3))
        z0 = int(lv.argmax()) if lv.any() else (S-n)//2
        z0 = max(0, min(z0-1, S-n))
        return im[:,:,z0:z0+n,:], seg[:,:,z0:z0+n,:]

    def remap(seg):
        out = np.zeros_like(seg)
        out[seg == 1] = 3  # RV  → RVBP
        out[seg == 2] = 2  # MYO → LVMYO
        out[seg == 3] = 1  # LV  → LVBP
        return out

    patient_dirs = sorted(glob.glob("/vol/raw/*/"))
    print(f"Found {len(patient_dirs)} patients")
    for pd in patient_dirs:
        pid = os.path.basename(pd.rstrip("/"))
        sa_path  = os.path.join(pd, "sa.nii.gz")
        seg_path = os.path.join(pd, "seg_sa.nii.gz")
        if not os.path.exists(sa_path) or not os.path.exists(seg_path):
            print(f"  SKIP {pid}: missing files")
            continue
        sa  = nib.load(sa_path).get_fdata().astype(np.float32)
        seg = nib.load(seg_path).get_fdata().astype(np.int32)
        seg = remap(seg)
        sa  = center_crop(sa,  TARGET_H, TARGET_W)
        seg = center_crop(seg, TARGET_H, TARGET_W)
        sa,  seg = select_slices(sa, seg, SAX_N)
        sa  = pad_time(sa,  TARGET_T)
        seg = pad_time(seg, TARGET_T)
        lax     = np.zeros((TARGET_H, TARGET_W, 3, TARGET_T), dtype=np.float32)
        seg_lax = np.zeros((TARGET_H, TARGET_W, 3, TARGET_T), dtype=np.float32)
        out_dir = f"/vol/processed/{pid}"
        os.makedirs(out_dir, exist_ok=True)
        np.savez(f"{out_dir}/processed_seg_allax.npz",
                 sax=sa, lax=lax, seg_sax=seg.astype(np.float32), seg_lax=seg_lax)
        print(f"  {pid}: sa={sa.shape} seg_unique={np.unique(seg)}")

    vol.commit()
    print("Done.")


@app.function(image=image, gpu=None, volumes={"/vol": vol})
def check_loaded_keys():
    import torch, sys
    sys.path.insert(0, "/vol/WholeHeartRL")
    ckpt = torch.load(
        "/vol/logs/checkpoints/run2/25-04-2026_02-20-05/model-epoch=194-val_PSNR=21.91.ckpt",
        weights_only=False,
        map_location="cpu"
    )
    keys = list(ckpt["state_dict"].keys())
    print(f"Total keys in checkpoint: {len(keys)}")
    for k in keys:
        if "enc_pos_embed" in k or "patch_embed" in k:
            print(f"  {k}: {ckpt['state_dict'][k].shape}")


@app.function(image=image, gpu=None, volumes={"/vol": vol})
def check_sample_shape():
    import sys, os
    sys.path.insert(0, "/vol/WholeHeartRL")
    _setup_environment()
    import numpy as np
    d = np.load("/vol/processed/054/processed_seg_allax.npz")
    print(f"sax shape: {d['sax'].shape}")
    print(f"seg_sax shape: {d['seg_sax'].shape}")
    
    # Simulate what dataset returns
    from data.datasets import Cardiac3DplusTSAX
    import pandas as pd
    from pathlib import Path
    paths = [Path("/vol/processed/054/processed_seg_allax.npz")]
    df = pd.DataFrame({"eid_87802": [54], "LVM (g)": [120.0]})
    dset = Cardiac3DplusTSAX(paths, df, "LVM (g)", load_seg=True, sax_slice_num=6)
    img, seg, idx = dset[0]
    print(f"dataset img shape: {img.shape}")
    print(f"dataset seg shape: {seg.shape}")



@app.function(image=image, gpu=None, volumes={"/vol": vol})
def check_val_dset_shape():
    import sys
    sys.path.insert(0, "/vol/WholeHeartRL")
    _setup_environment()
    import pickle
    import pandas as pd
    from pathlib import Path
    from data.datasets import Cardiac3DplusTSAX_Test

    with open("/vol/dataloader/cmr_subject_paths.pkl", "rb") as f:
        paths = pickle.load(f)
    df = pd.read_pickle("/vol/dataloader/biomarker_table.pkl")

    val_dset = Cardiac3DplusTSAX_Test(
        paths["val"], df, "LVM (g)",
        load_seg=True, sax_slice_num=6
    )
    img, seg, idx = val_dset[0]
    print(f"val_dset[0] img shape: {img.shape}")
    print(f"val_dset[0] seg shape: {seg.shape}")
    print(f"val_dset.num_classes: {val_dset.num_classes}")
    print(f"val_dset.view: {val_dset.view}")



@app.function(image=image, gpu=None, volumes={"/vol": vol})
def check_configs():
    for name in ["config_segmentation_acdc_pretrained.yaml",
                 "config_segmentation_acdc_scratch.yaml"]:
        print(f"\n--- {name} ---")
        with open(f"/vol/WholeHeartRL/configs/{name}") as f:
            print(f.read())

@app.function(image=image, gpu=None, volumes={"/vol": vol})
def setup_volume_dirs():
    import os
    for d in ["/vol/tabular", "/vol/raw", "/vol/logs"]:
        os.makedirs(d, exist_ok=True)
    vol.commit()
    print("Done")

@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/vol": vol})
def evaluate(condition: str):
    import sys, os
    sys.path.insert(0, "/vol/WholeHeartRL")
    _setup_environment()
    import torch
    import numpy as np
    import pickle
    import pandas as pd
    from pathlib import Path
    from data.datasets import Cardiac3DplusTSAX_Test
    from torch.utils.data import DataLoader

    ckpt_map = {
        "scratch":    "/vol/logs/checkpoints/seg_scratch/25-04-2026_10-28-32/model-epoch=204-val_Dice_FG=0.71.ckpt",
        "pretrained": "/vol/logs/checkpoints/seg_pretrained/25-04-2026_10-24-54/model-epoch=299-val_Dice_FG=0.67.ckpt",
    }

    # Load test dataset
    with open("/vol/dataloader/cmr_subject_paths.pkl", "rb") as f:
        paths = pickle.load(f)
    df = pd.read_pickle("/vol/dataloader/biomarker_table.pkl")
    test_dset = Cardiac3DplusTSAX_Test(
        paths["test"], df, "LVM (g)", load_seg=True, sax_slice_num=6
    )
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0)

    # Load model
    from models.segmentation_models import SegMAE
    ckpt = torch.load(ckpt_map[condition], weights_only=False, map_location="cpu")

    # Build model from hparams
    hparams = ckpt["hyper_parameters"]
    hparams.pop("val_dset", None)
    model = SegMAE(val_dset=test_dset, **hparams)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().cuda()

    # Run evaluation
    from utils.general import to_1hot
    num_classes = test_dset.num_classes
    dice_per_class = torch.zeros(num_classes)
    count = 0

    with torch.no_grad():
        for imgs, segs, idx in test_loader:
            imgs = imgs.cuda()
            segs = segs.cuda()
            pred = model(imgs)  # (B, C, S, T, H, W)
            pred_hard = torch.argmax(pred, dim=1)  # (B, S, T, H, W)

            gt = to_1hot(segs, num_class=num_classes).moveaxis(-1, 1)  # (B, C, S, T, H, W)
            pred_1hot = to_1hot(pred_hard, num_class=num_classes).moveaxis(-1, 1)

            for c in range(num_classes):
                intersection = (pred_1hot[:, c] * gt[:, c]).sum()
                denom = pred_1hot[:, c].sum() + gt[:, c].sum()
                dice = (2 * intersection / denom) if denom > 0 else torch.tensor(1.0)
                dice_per_class[c] += dice.item()
            count += 1

    dice_per_class /= count
    iou_per_class = dice_per_class / (2 - dice_per_class)

    print(f"\n=== TEST RESULTS — {condition.upper()} ===")
    classes = ["BG", "LVBP", "LVMYO", "RVBP"]
    for c, name in enumerate(classes):
        print(f"  {name}: Dice={dice_per_class[c]:.4f}  IoU={iou_per_class[c]:.4f}")
    fg_dice = dice_per_class[1:].mean()
    fg_iou = iou_per_class[1:].mean()
    print(f"  FG mean: Dice={fg_dice:.4f}  IoU={fg_iou:.4f}")

@app.local_entrypoint()
def main(condition: str = "both"):
    """
    Run segmentation finetuning.

    Usage:
        modal run modal_seg.py                         # both conditions
        modal run modal_seg.py --condition pretrained
        modal run modal_seg.py --condition scratch
    """
    assert condition in ("pretrained", "scratch", "pretrained_20", "scratch_20", "both"), \
        f"--condition must be pretrained, scratch, or both. Got: {condition}"

    conditions = ["pretrained", "scratch"] if condition == "both" else [condition]

    for cond in conditions:
        print(f"\n>>> Launching {cond} condition on Modal...")
        code = train_segmentation.remote(cond)
        if code != 0:
            print(f"ERROR: {cond} run failed with exit code {code}. Check Modal logs.")
        else:
            print(f">>> {cond} run completed successfully.")
