# Self-Supervised Representation Learning for Cardiac MRI Segmentation

Research project investigating cross-domain transfer of self-supervised cardiac MRI representations to Late-Gadolinium Enhancement (LGE) segmentation.

**Supervisors:** Dr. Karen Sanchez, Pr. Bernard Ghanem  
**Institution:** King Abdullah University of Science and Technology (KAUST)  
**Target Venue:** CVPR / MICCAI

---

## Overview

This project adapts [WholeHeartRL](https://github.com/Yundi-Zhang/WholeHeartRL) (MICCAI 2024, Oral & Best Paper Nominee) — a masked autoencoder framework for cardiac MRI representation learning — and evaluates its transfer to myocardial infarct segmentation from LGE-MRI.

**Research Questions:**
- Can self-supervised representations learned from unlabeled cardiac cine MRI transfer to LGE-MRI segmentation despite the domain gap?
- Does SSL pretraining disproportionately benefit rare, small structures (infarct, MVO)?
- How does performance degrade under limited annotation budgets?

## Project Structure

```
cardiac-project/
├── WholeHeartRL/              # Adapted WholeHeartRL codebase (patched for ACDC)
│   ├── configs/               # Training configs (reconstruction, segmentation, regression)
│   ├── data/                  # Dataloaders and dataset classes
│   ├── models/                # MAE reconstruction, segmentation, regression models
│   ├── networks/              # ViT encoder, UNETR decoder, losses
│   └── utils/                 # Data processing, logging, model utilities
├── convert_acdc.py            # Converts ACDC dataset to WholeHeartRL format
├── create_dummy_tabular.py    # Creates placeholder tabular data for ACDC
├── prepare_pipeline_data.py   # Preprocesses data into .npz format with train/val/test splits
├── modal_run.py               # Modal cloud GPU training script (pretraining)
├── modal_seg.py               # Modal cloud GPU training script (segmentation)
└── dot_env_file.txt           # Environment variable template
```

## Datasets

| Dataset | Modality | Cases | Usage | Access |
|---------|----------|-------|-------|--------|
| [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/) | Cardiac cine MRI | 100 | Proxy for pipeline development | Public |
| [UK Biobank CMR](https://www.ukbiobank.ac.uk/) | Cardiac cine MRI | ~14,000 | SSL pretraining | Requires application |
| [MyoSAIQ](https://www.creatis.insa-lyon.fr/Challenge/myosaiq/tasks.html) | LGE-MRI | 439 | Infarct segmentation evaluation | Requires registration |

## Modifications to WholeHeartRL

The original codebase was designed for UK Biobank. Key adaptations for ACDC:

- **`utils/data_related.py`**: Removed hardcoded hostname detection; replaced with environment variable-based path configuration
- **`data/dataloaders.py`**: Added path remapping for cross-platform compatibility (Windows → Linux)
- **`configs/`**: Adjusted patch size from `[25, 8, 8]` to `[5, 8, 8]` (ACDC has 20 standardized timeframes vs UKBB's 50), reduced training scale
- **`main.py`**: Disabled Weights & Biases online logging for cloud execution
- **Data pipeline**: Created conversion scripts to transform ACDC NIfTI format to WholeHeartRL's expected directory structure with dummy long-axis views

## Setup & Reproduction

### Prerequisites
- Python 3.11+
- [Modal](https://modal.com/) account (for cloud GPU training) or local NVIDIA GPU
- ACDC dataset downloaded

### Step 1: Convert ACDC Data
```bash
pip install nibabel numpy pandas
python convert_acdc.py --acdc_dir ./ACDC/database/training --output_dir ./data/raw
```

### Step 2: Prepare Pipeline Data
```bash
python prepare_pipeline_data.py --data_dir ./data/raw --output_dir ./data
```

### Step 3: Run SSL Pretraining (Modal)
```bash
pip install modal
modal setup
modal volume create cardiac-data
modal volume put cardiac-data data/raw /raw
modal volume put cardiac-data data/tabular /tabular
modal volume put cardiac-data data/dataloader /dataloader
modal volume put cardiac-data data/processed /processed
modal volume put cardiac-data WholeHeartRL /WholeHeartRL
modal run modal_run.py
```

### Step 4: Run Segmentation Finetuning
```bash
modal run modal_seg.py
```

### Step 5: Download Results
```bash
modal volume get cardiac-data /logs ./logs
```

## Current Results

### SSL Pretraining (Phase 1)
| Metric | Value |
|--------|-------|
| Dataset | ACDC (70 train / 15 val / 15 test) |
| Epochs | 50 |
| Mask ratio | 0.7 |
| Patch size | [5, 8, 8] |
| Best val PSNR | **19.78** (epoch 34) |
| GPU | NVIDIA A10G (Modal) |

### Segmentation (Phase 2)
*In progress*

## Planned Experiments

| Experiment | Data | Method | Status |
|------------|------|--------|--------|
| SSL Pretraining on ACDC | ACDC (100) | Masked Autoencoder | ✅ Complete |
| Segmentation finetuning on ACDC | ACDC (100) | MAE encoder + UNETR decoder | 🔄 In progress |
| U-Net baseline on MyoSAIQ | MyoSAIQ (358) | 3D U-Net (Dice+CE) | ⬜ Pending data |
| SSL Pretraining on UKBB | UKBB (~14k) | Masked Autoencoder | ⬜ Pending access |
| Transfer: UKBB → MyoSAIQ | UKBB + MyoSAIQ | Pretrained encoder + finetune | ⬜ Pending |
| Label efficiency study | MyoSAIQ subsets | 10%/25%/50%/100% labels | ⬜ Pending |

## References

1. Zhang et al., "Whole Heart 3D+T Representation Learning Through Sparse 2D Cardiac MR Images," MICCAI 2024.
2. FM-ABS, "Promptable Foundation Model Drives Active Barely Supervised Learning," MICCAI 2024.
3. He et al., "VISTA3D: A Unified Segmentation Foundation Model," CVPR 2025.
4. Gao et al., "Training Like a Medical Resident: Context-Prior Learning," CVPR 2024.
5. MyoSAIQ Challenge, CREATIS, Lyon.

## License

This project builds on [WholeHeartRL](https://github.com/Yundi-Zhang/WholeHeartRL) (MIT License).
