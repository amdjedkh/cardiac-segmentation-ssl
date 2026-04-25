# Self-Supervised Representation Learning for Cardiac MRI Segmentation

This repository adapts [WholeHeartRL](https://github.com/Yundi-Zhang/WholeHeartRL) (MICCAI 2024) — a masked autoencoder (MAE) framework for cardiac MRI representation learning — and evaluates whether SSL pretraining improves downstream segmentation compared to training from scratch. The pipeline is validated on ACDC and is designed to extend to myocardial infarct segmentation on LGE-MRI (MyoSAIQ).

---

## Research Questions

- Does MAE-based SSL pretraining on cardiac cine MRI improve downstream segmentation over random initialization?
- Can representations learned from cine MRI transfer to LGE-MRI segmentation despite the domain gap?
- Does SSL pretraining disproportionately benefit rare, small structures (infarct, MVO) under limited annotation budgets?

---

## Repository Structure

```
├── WholeHeartRL/
│   ├── configs/
│   │   ├── config_reconstruction.yaml                # MAE pretraining
│   │   ├── config_segmentation_acdc_pretrained.yaml  # Finetuning — pretrained encoder
│   │   └── config_segmentation_acdc_scratch.yaml     # Finetuning — random init
│   ├── data/                   # Dataset classes and dataloaders
│   ├── models/                 # ReconMAE, SegMAE, RegrMAE
│   ├── networks/               # ViT encoder, UNETR decoder, losses
│   └── utils/                  # Preprocessing, logging, model utilities
├── convert_acdc.py             # Convert ACDC NIfTI → WholeHeartRL npz format
├── create_dummy_tabular.py     # Generate placeholder tabular files for ACDC
├── prepare_pipeline_data.py    # Build train/val/test splits and pickle files
├── modal_run.py                # Cloud training script — MAE pretraining (Modal)
├── modal_seg.py                # Cloud training script — segmentation finetuning + evaluation
└── plot_curves.py              # Learning curve visualization
```

---

## Datasets

| Dataset | Modality | Subjects | Role |
|---------|----------|----------|------|
| [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | Cine MRI (SA) | 100 | Pretraining + segmentation finetuning |
| [UK Biobank](https://www.ukbiobank.ac.uk/) | Cine MRI | ~14,000 | Large-scale pretraining (pending access) |
| [MyoSAIQ](https://www.creatis.insa-lyon.fr/Challenge/myosaiq/) | LGE-MRI | 439 | Infarct segmentation transfer (pending access) |

---

## Modifications to WholeHeartRL

The original codebase targets UK Biobank exclusively. The following changes were made to support ACDC and ensure reproducibility:

| File | Change |
|------|--------|
| `main.py` | `torch.load(..., weights_only=False)` for PyTorch 2.6 compatibility |
| `main.py` | Encoder weight transfer includes `enc_pos_embed` and `patch_embed` |
| `main.py` | W&B logging disabled for cloud execution |
| `models/segmentation_models.py` | Fixed `AttributeError` on missing `vis` in `test_step` |
| `utils/params.py` | Added `precision` field to `TrainerParams` |
| `utils/data_related.py` | Replaced hardcoded hostname detection with env variable paths |
| `data/dataloaders.py` | Added path remapping for cross-platform compatibility |
| `configs/` | New ACDC configs: `patch_size=[1,8,8]`, `enc_embed_dim=1040`, SA-only, T=2 |

---

## Setup

### Requirements

- Python 3.11+
- [Modal](https://modal.com/) account (cloud GPU) or local NVIDIA GPU with ≥24 GB VRAM
- ACDC dataset: download from https://www.creatis.insa-lyon.fr/Challenge/acdc/

### Installation

```bash
git clone https://github.com/amdjedkh/cardiac-segmentation-ssl
cd cardiac-segmentation-ssl
pip install modal nibabel numpy pandas matplotlib torch torchvision
```

---

## Reproduction

### Step 1 — Upload data to Modal volume

```bash
modal setup
modal volume create cardiac-data
modal volume put cardiac-data <path-to-acdc>/training raw --force
modal volume put cardiac-data WholeHeartRL WholeHeartRL --force
```

### Step 2 — Rebuild processed npz files

```bash
modal run modal_seg.py::rebuild_npz_2frame
```

Produces `/vol/processed/<id>/processed_seg_allax.npz` per subject:
- `sax`: `(128, 128, 6, 2)` — 6 SA slices, ED and ES frames
- `seg_sax`: `(128, 128, 6, 2)` — labels: 1=LVBP, 2=LVMYO, 3=RVBP
- `lax`, `seg_lax`: zeros (ACDC has no LA annotations)

### Step 3 — Generate data splits

```bash
modal run modal_seg.py::create_acdc_pickles
```

Fixed split: 70 train / 15 val / 15 test (seed=1).

### Step 4 — MAE pretraining

```bash
modal run --detach modal_run.py
```

Checkpoints saved to `/vol/logs/checkpoints/run2/`.

### Step 5 — Segmentation finetuning

```bash
modal run --detach modal_seg.py                        # both conditions
modal run --detach modal_seg.py --condition pretrained
modal run --detach modal_seg.py --condition scratch
```

### Step 6 — Test evaluation

```bash
modal run modal_seg.py::evaluate --condition scratch
modal run modal_seg.py::evaluate --condition pretrained
```

### Step 7 — Download logs and plot learning curves

```bash
modal volume get cardiac-data logs/lightning_logs .
python plot_curves.py \
  --scratch lightning_logs/version_16/metrics.csv \
  --pretrained lightning_logs/version_19/metrics.csv \
  --output learning_curves.png
```

---

## Results

### MAE Pretraining

| Setting | Value |
|---------|-------|
| Dataset | ACDC — 70 training subjects |
| Input | SA-only, ED + ES frames (T=2) |
| Patch size | [1, 8, 8] |
| enc_embed_dim | 1040 |
| Mask ratio | 0.7 |
| Epochs | 200 |
| Best val PSNR | **21.91** (epoch 194) |

### Segmentation Finetuning — ACDC Test Set

Identical architecture and training protocol across both conditions. Only `load_encoder` differs.

| Condition | LVBP Dice | LVMYO Dice | RVBP Dice | FG Dice | FG IoU |
|-----------|-----------|------------|-----------|---------|--------|
| Scratch | 0.761 | 0.355 | 0.356 | 0.491 | 0.349 |
| Pretrained (SSL) | 0.735 | 0.358 | 0.358 | 0.484 | 0.339 |

The pretrained condition converges faster in early epochs (epoch 5: 0.194 vs 0.169 Dice FG) but reaches equivalent final performance at epoch 100. This suggests MAE pretraining provides a useful initialization but its advantage diminishes with sufficient supervised training on this dataset scale.


---

## References

1. Zhang et al., "Whole Heart 3D+T Representation Learning Through Sparse 2D Cardiac MR Images," MICCAI 2024.
2. He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022.
3. He et al., "VISTA3D: A Unified Segmentation Foundation Model," CVPR 2025.
4. Gao et al., "Training Like a Medical Resident: Context-Prior Learning Toward Universal Medical Image Segmentation," CVPR 2024.
5. MyoSAIQ Challenge, CREATIS, Lyon.

---

## License

Builds on [WholeHeartRL](https://github.com/Yundi-Zhang/WholeHeartRL) (MIT License).
