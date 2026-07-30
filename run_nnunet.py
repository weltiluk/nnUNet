#!/usr/bin/env python3
"""Plan, preprocess, train and predict a prepared nnU-Net dataset."""
from __future__ import annotations
import argparse, os, re, shutil, subprocess
from datetime import datetime
from pathlib import Path

""" C-compiler installieren: (und tmux)
sudo apt update && sudo apt install -y build-essential && sudo apt install -y python3.12-dev && sudo apt install -y tmux
"""

"""
python run_nnunet.py /workspaces/l40-workspace/nnUNet/results/real_dataset_splits_70_15_15/raw/Dataset501_real_dataset_splits_70_15_15 \
&& python run_nnunet.py /workspaces/l40-workspace/nnUNet/results/r4_s1_70_15_15_brain_prior/raw/Dataset501_r4_s1_70_15_15_brain_prior
"""

TRAINER = "nnUNetTrainerDiceEarlyStoppingTensorboard"
batch_size = 4  # mit 8 bricht workspace ab bevor epoche 1 startet

def execute(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_dataset", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default=TRAINER)
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--run-name", help="run folder name (default: current timestamp)")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-prediction", action="store_true")
    args = parser.parse_args()
    dataset = args.prepared_dataset.resolve()
    if not (dataset / "dataset.json").is_file():
        raise FileNotFoundError(f"Not a prepared dataset: {dataset}")
    match = re.fullmatch(r"Dataset(\d{3})_.+", dataset.name)
    if not match:
        raise ValueError("Prepared folder must be named DatasetXXX_Name")
    dataset_id = match.group(1)
    workspace = dataset.parent.parent
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError("--run-name may only contain letters, numbers, dot, underscore and hyphen")
    preprocessed = workspace / "preprocessed"
    results = workspace / "training" / run_name
    preprocessed.mkdir(exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be an integer >= 1")
    env = os.environ.copy()
    env["NNUNET_BATCH_SIZE"] = str(batch_size)
    env.update(nnUNet_raw=str(dataset.parent), nnUNet_preprocessed=str(preprocessed), nnUNet_results=str(results))
    if not args.skip_preprocessing:
        execute(["nnUNetv2_plan_and_preprocess", "-d", dataset_id, "--verify_dataset_integrity"], env)
    prepared_target = preprocessed / dataset.name
    prepared_target.mkdir(exist_ok=True)
    shutil.copy2(dataset / "splits_final.json", prepared_target / "splits_final.json")
    common = ["-tr", args.trainer, "-p", args.plans]
    if not args.skip_training:
        execute(["nnUNetv2_train", dataset_id, args.configuration, str(args.fold), *common, "--val_best"], env)
    images_ts = dataset / "imagesTs"
    if not args.skip_prediction and images_ts.is_dir() and any(images_ts.glob("*.nii.gz")):
        predictions = workspace / "predictions" / run_name
        predictions.mkdir(parents=True, exist_ok=True)
        execute(["nnUNetv2_predict", "-i", str(images_ts), "-o", str(predictions), "-d", dataset_id, "-c", args.configuration, "-f", str(args.fold), *common, "-chk", "checkpoint_best.pth"], env)
        print(f"Predictions: {predictions}")
    else:
        print("Prediction skipped (no test images or explicitly disabled).")
    print(f"Training run: {results}")

if __name__ == "__main__":
    main()
