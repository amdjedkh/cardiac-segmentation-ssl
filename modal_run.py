import modal

app = modal.App("wholeheart-rl")

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


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 4,
    volumes={"/vol": vol},
)
def train_pretrain():
    import subprocess
    import os
    import sys

    # Set environment variables
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
    os.makedirs("/vol/processed", exist_ok=True)

    # Add code to Python path
    sys.path.insert(0, "/vol/WholeHeartRL")

    # Verify setup
    print("=== SETUP CHECK ===")
    print(f"Processed data: {os.listdir('/vol/processed')[:5]}...")
    print(f"Tabular: {os.listdir('/vol/tabular') if os.path.exists('/vol/tabular') else 'not needed for pretraining'}")
    print(f"Dataloader: {os.listdir('/vol/dataloader')}")

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=== STARTING TRAINING ===")
    # Patch Lightning for PyTorch 2.6
    filepath = "/usr/local/lib/python3.11/site-packages/lightning/fabric/utilities/cloud_io.py"
    with open(filepath, "r") as f:
        lines = f.readlines()
    new_lines = [l.replace("weights_only=weights_only,", "weights_only=False,") for l in lines]
    with open(filepath, "w") as f:
        f.writelines(new_lines)
    # Run training
    result = subprocess.run(
        [
            sys.executable, "/vol/WholeHeartRL/main.py", "train",
            "-c", "/vol/WholeHeartRL/configs/config_reconstruction.yaml",
            "-g", "pretrain_acdc",
            "-n", "run2",
        ],
        cwd="/vol/WholeHeartRL",
        env=os.environ.copy(),
        capture_output=False,
    )

    print(f"\n=== FINISHED WITH CODE: {result.returncode} ===")
    vol.commit()
    return result.returncode


@app.local_entrypoint()
def main():
    print("Starting WholeHeartRL pretraining on Modal...")
    code = train_pretrain.remote()
    print(f"Done. Exit code: {code}")
