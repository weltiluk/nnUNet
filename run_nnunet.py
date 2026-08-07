#!/usr/bin/env python3
"""Plan, preprocess, train and predict a prepared nnU-Net dataset."""
from __future__ import annotations
import argparse, json, math, os, re, shutil, subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt

""" C-compiler installieren: (und tmux)
sudo apt update && sudo apt install -y build-essential && sudo apt install -y python3.12-dev && sudo apt install -y tmux
"""

"""
python run_nnunet.py /workspaces/l40-workspace/nnUNet/results/real_dataset_splits_70_15_15/raw/Dataset501_real_dataset_splits_70_15_15


python run_nnunet.py \
  results/real_dataset_splits_70_15_15/raw/Dataset501_real_dataset_splits_70_15_15 \
  --run-name real_dataset_splits_70_15_15_run2 \
&& python run_nnunet.py \
  results/r4_s1_70_15_15_brain_transform_image_order1/raw/Dataset501_r4_s1_70_15_15_brain_transform_image_order1 \
  --run-name r4_s1_70_15_15_brain_transform_image_order1_run2 \
&& python run_nnunet.py \
  results/r4_s1_70_15_15_brain_prior/raw/Dataset501_r4_s1_70_15_15_brain_prior \
  --run-name r4_s1_70_15_15_brain_prior_run2

"""

""" nur test metriken berechnen
python run_nnunet.py \
  /workspaces/l40-workspace/nnUNet/results/r4_s1_70_15_15_brain_prior/raw/Dataset501_r4_s1_70_15_15_brain_prior \
  --run-name 20260802_105902 \
  --skip-preprocessing \
  --skip-training \
  --skip-prediction
"""


TRAINER = "nnUNetTrainerDiceEarlyStoppingTensorboard"
batch_size = 4  # mit 8 bricht workspace ab bevor epoche 1 startet

def execute(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)

def _surface(mask: np.ndarray) -> np.ndarray:
    """Return the inner surface of a 2D or 3D binary mask."""
    return mask & ~binary_erosion(mask, border_value=0)


def hd95_voxels(mask_ref: np.ndarray, mask_pred: np.ndarray) -> float:
    """Symmetric HD95 in voxels, with explicit handling of empty masks."""
    mask_ref = np.asarray(mask_ref, dtype=bool)
    mask_pred = np.asarray(mask_pred, dtype=bool)
    if mask_ref.shape != mask_pred.shape:
        raise ValueError(f"Reference and prediction shapes differ: {mask_ref.shape} != {mask_pred.shape}")

    ref_present = bool(mask_ref.any())
    pred_present = bool(mask_pred.any())
    if not ref_present and not pred_present:
        return math.nan
    if ref_present != pred_present:
        # Same convention as the existing 2D evaluation, generalized to 3D.
        return math.sqrt(sum(length ** 2 for length in mask_ref.shape))

    surface_ref = _surface(mask_ref)
    surface_pred = _surface(mask_pred)
    ref_to_pred = distance_transform_edt(~surface_pred)[surface_ref]
    pred_to_ref = distance_transform_edt(~surface_ref)[surface_pred]
    # MONAI-style symmetric reduction: percentile per direction, then maximum.
    return float(max(np.percentile(ref_to_pred, 95), np.percentile(pred_to_ref, 95)))


def add_test_metrics(summary_file: Path) -> None:
    """Add precision, recall and voxel-based HD95 to the nnU-Net test summary."""
    with summary_file.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    for case in summary["metric_per_case"]:
        reference = sitk.GetArrayFromImage(sitk.ReadImage(case["reference_file"]))
        prediction = sitk.GetArrayFromImage(sitk.ReadImage(case["prediction_file"]))
        if reference.shape != prediction.shape:
            raise ValueError(
                f"Reference and prediction shapes differ for {case['prediction_file']}: "
                f"{reference.shape} != {prediction.shape}"
            )
        for label, metrics in case["metrics"].items():
            precision_denominator = metrics["TP"] + metrics["FP"]
            recall_denominator = metrics["TP"] + metrics["FN"]
            metrics["Precision"] = metrics["TP"] / precision_denominator if precision_denominator else math.nan
            metrics["Recall"] = metrics["TP"] / recall_denominator if recall_denominator else math.nan
            metrics["HD95"] = hd95_voxels(reference == int(label), prediction == int(label))

    for label, mean_metrics in summary["mean"].items():
        for metric_name in ("Precision", "Recall", "HD95"):
            values = [
                case["metrics"][label][metric_name]
                for case in summary["metric_per_case"]
                if not math.isnan(case["metrics"][label][metric_name])
            ]
            mean_metrics[metric_name] = sum(values) / len(values) if values else math.nan

    per_class = summary["mean"]
    for metric_name in ("Precision", "Recall", "HD95"):
        values = [
            metrics[metric_name]
            for label, metrics in per_class.items()
            if label != "0"
        ]
        values = [value for value in values if not math.isnan(value)]
        summary["foreground_mean"][metric_name] = sum(values) / len(values) if values else math.nan

    overall_mean = {}
    metric_names = {name for metrics in per_class.values() for name in metrics}
    for metric_name in metric_names:
        if metric_name == "HD95":
            continue
        values = [metrics[metric_name] for metrics in per_class.values() if metric_name in metrics]
        values = [value for value in values if not math.isnan(value)]
        overall_mean[metric_name] = sum(values) / len(values) if values else math.nan

    summary = {
        "foreground_mean": summary["foreground_mean"],
        "mean": {"all_classes": overall_mean, **per_class},
        "metric_per_case": summary["metric_per_case"],
    }

    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=4)
        handle.write("\n")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared_dataset", type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default=TRAINER)
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--run-name", help="run folder name (default: current timestamp)")
    parser.add_argument("--evaluation-processes", type=int, default=2)
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
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
    if args.evaluation_processes < 1:
        raise ValueError("--evaluation-processes must be >= 1")
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
    labels_ts = dataset / "labelsTs"
    predictions = workspace / "predictions" / run_name
    if not args.skip_prediction and images_ts.is_dir() and any(images_ts.glob("*.nii.gz")):
        predictions.mkdir(parents=True, exist_ok=True)
        execute(["nnUNetv2_predict", "-i", str(images_ts), "-o", str(predictions), "-d", dataset_id, "-c", args.configuration, "-f", str(args.fold), *common, "-chk", "checkpoint_best.pth"], env)
        print(f"Predictions: {predictions}")
    else:
        print("Prediction skipped (no test images or explicitly disabled).")
    if not args.skip_evaluation:
        if not labels_ts.is_dir() or not any(labels_ts.glob("*.nii.gz")):
            print("Evaluation skipped (no test labels).")
        elif not predictions.is_dir() or not any(predictions.glob("*.nii.gz")):
            print("Evaluation skipped (no test predictions).")
        else:
            dataset_json = predictions / "dataset.json"
            plans_json = predictions / "plans.json"
            if not dataset_json.is_file() or not plans_json.is_file():
                raise FileNotFoundError(
                    f"Prediction metadata missing: {dataset_json} or {plans_json}"
                )
            execute([
                "nnUNetv2_evaluate_folder",
                str(labels_ts),
                str(predictions),
                "-djfile", str(dataset_json),
                "-pfile", str(plans_json),
                "-o", str(predictions / "summary.json"),
                "-np", str(args.evaluation_processes),
                "--include-background",
            ], env)
            summary_file = predictions / "summary.json"
            add_test_metrics(summary_file)
            print(f"Test metrics: {summary_file}")
    print(f"Training run: {results}")

if __name__ == "__main__":
    main()
