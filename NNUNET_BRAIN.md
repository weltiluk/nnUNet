# nnU-Net-Workflow für die 3D-NIfTI-Daten

Eine `.md`-Datei ist nur Dokumentation und wird nicht ausgeführt.

## 1. Dataset vorbereiten

Der Eingabeordner enthält `train.csv`, `validation.csv` und optional
`test.csv`. Jede CSV enthält `image_path` und `mask_path` für die 3D-NIfTI-
Dateien (`.nii.gz`).

Vom Workspace-Verzeichnis aus:

```bash
cd /workspaces/masterarbeit
python nnUNet/prepare_for_nnunet.py /pfad/zum/dataset
```

Das Skript verwendet automatisch Hintergrund `0` sowie `class_1=1`,
`class_2=2` und `class_3=3`. Es gibt anschließend den vollständigen Folgeaufruf
aus. Falls Dataset-ID 501 bereits verwendet wird, kann beim Prepare-Aufruf
beispielsweise `--dataset-id 502` ergänzt werden.

## 2. Preprocessing, Training und Testsegmentierung

Den ausgegebenen Befehl direkt ausführen, beispielsweise:

```bash
python nnUNet/run_nnunet.py \
  /workspaces/masterarbeit/nnUNet/results/DATASETNAME/raw/Dataset501_DATASETNAME
```

Optional kann der Lauf benannt werden:

```bash
python nnUNet/run_nnunet.py /pfad/zum/prepared-dataset \
  --run-name baseline_fold0
```

Ohne `--run-name` wird ein Zeitstempel wie `20260727_130500` verwendet. Der
Runner erledigt Dataset-Prüfung, Planung, Preprocessing, Split-Übernahme,
Training und die Segmentierung der Testbilder mit `checkpoint_best.pth`.

## Ergebnisstruktur

Alle erzeugten Daten liegen zentral im nnUNet-Projekt:

```text
nnUNet/results/
└── DATASETNAME/
    ├── raw/
    │   └── Dataset501_DATASETNAME/
    ├── preprocessed/
    ├── training/
    │   ├── 20260727_130500/
    │   └── baseline_fold0/
    └── predictions/
        ├── 20260727_130500/
        └── baseline_fold0/
```

TensorBoard kann aus `/workspaces/masterarbeit/nnUNet` über alle Datasets und
Runs gestartet werden:

```bash
cd /workspaces/masterarbeit/nnUNet
tensorboard --logdir results
```
