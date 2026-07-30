#!/usr/bin/env python3
"""Convert a CSV-split 3D NIfTI dataset to nnU-Net v2 raw format."""
from __future__ import annotations
import argparse, csv, json, re, shutil
from dataclasses import dataclass
from pathlib import Path

"""
python prepare_for_nnunet.py /workspaces/l40-workspace/welti-masterarbeit/real_dataset_splits_70_15_15
python prepare_for_nnunet.py /workspaces/l40-workspace/welti-masterarbeit/mixed_datasets/r3_s1_70_15_15_brain_transform_image
"""

SUFFIX = ".nii.gz"
LABELS = {"background": 0, "class_1": 1, "class_2": 2, "class_3": 3}

@dataclass(frozen=True)
class Case:
    identifier: str
    image: Path
    mask: Path
    split: str

def stem(path: Path) -> str:
    return path.name[:-len(SUFFIX)]

def safe_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    if not value:
        raise ValueError("Could not construct a case ID")
    return value

def resolve(value: str, csv_path: Path, split: str, folder: str) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [
        csv_path.parent / path,
        csv_path.parent / split / folder / path.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()

def read_split(root: Path, split: str, optional: bool = False) -> list[Case]:
    csv_path = root / f"{split}.csv"
    if not csv_path.is_file():
        if optional:
            return []
        raise FileNotFoundError(csv_path)
    cases = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"image_path", "mask_path"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} needs image_path and mask_path")
        for line, row in enumerate(reader, 2):
            if not row["image_path"].strip() or not row["mask_path"].strip():
                raise ValueError(f"{csv_path}:{line}: image_path/mask_path is empty")
            image = resolve(row["image_path"].strip(), csv_path, split, "images")
            mask = resolve(row["mask_path"].strip(), csv_path, split, "masks")
            cases.append(Case(safe_id(stem(image)), image, mask, split))
    return cases

def validate(cases: list[Case]) -> None:
    if not cases:
        raise ValueError("No cases found")
    seen = set()
    for case in cases:
        if case.identifier in seen:
            raise ValueError(f"Duplicate case ID: {case.identifier}")
        seen.add(case.identifier)
        for path in (case.image, case.mask):
            if not path.is_file():
                raise FileNotFoundError(path)
            if not path.name.endswith(SUFFIX):
                raise ValueError(f"Only .nii.gz is supported: {path}")

def transfer(source: Path, target: Path, copy: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    shutil.copy2(source, target) if copy else target.symlink_to(source.resolve())

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.dataset_id <= 999:
        raise ValueError("--dataset-id must be between 1 and 999")
    root = args.dataset_dir.resolve()
    cases = read_split(root, "train") + read_split(root, "validation") + read_split(root, "test", True)
    validate(cases)
    output = args.output.resolve() if args.output else Path(__file__).resolve().parent / "results" / safe_id(root.name) / "raw" / f"Dataset{args.dataset_id:03d}_{safe_id(root.name)}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if case.split == "test":
            transfer(case.image, output / "imagesTs" / f"{case.identifier}_0000.nii.gz", args.copy)
            transfer(case.mask, output / "labelsTs" / f"{case.identifier}.nii.gz", args.copy)
        else:
            transfer(case.image, output / "imagesTr" / f"{case.identifier}_0000.nii.gz", args.copy)
            transfer(case.mask, output / "labelsTr" / f"{case.identifier}.nii.gz", args.copy)
    training = [c for c in cases if c.split != "test"]
    dataset_json = {"name": output.name, "channel_names": {"0": "image"}, "labels": LABELS, "numTraining": len(training), "file_ending": SUFFIX}
    (output / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")
    splits = [{"train": [c.identifier for c in cases if c.split == "train"], "val": [c.identifier for c in cases if c.split == "validation"]}]
    (output / "splits_final.json").write_text(json.dumps(splits, indent=2) + "\n")
    print(f"Prepared dataset: {output}")
    print(f"Next: python run_nnunet.py {output}")

if __name__ == "__main__":
    main()
