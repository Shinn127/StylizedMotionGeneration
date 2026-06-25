# StylizedMotionGeneration

> Style-Controllable Motion Generation

## Overview

This project focuses on stylized motion generation — synthesizing motion sequences
with controllable stylistic / emotional characteristics (e.g. angry, happy, sad,
old, etc.), typically built on top of motion diffusion or auto-regressive
frameworks.

## Environment

This project uses a conda environment named `mcc`.

```bash
conda activate mcc
```

Python: see `environment.yml` / `requirements.txt`.

## Project Layout

```
StylizedMotionGeneration/
├── README.md
├── .gitignore
├── requirements.txt      # pip dependencies
├── environment.yml       # conda environment spec (optional)
├── data/                 # local symlinks + processed outputs (git-ignored)
├── preprocess/           # BVH/database preprocessing scripts
├── configs/              # training / inference configs
├── models/               # model definitions
├── train.py
├── generate.py
└── utils/
```

## Dataset Setup

Create local symlinks under `data/raw/`:

```bash
ln -s /path/to/lafan data/raw/lafan
ln -s /path/to/100style data/raw/100style
```

Build a database in the style of `ControlOperators-main`:

```bash
python preprocess/build_database.py --dataset lafan
python preprocess/build_database.py --dataset 100style
python preprocess/build_database.py --dataset 100style --max-styles 5 --output data/processed/100style_test5
python preprocess/build_database.py --dataset 100style --max-styles 5 --prune-ends-and-fingers --output data/processed/100style_test5_pruned
```

Outputs are written to `data/processed/<dataset>/database.npz`.

Build ControlOperators-style trajectory inputs from an existing database:

```bash
python preprocess/build_trajectory_inputs.py --dataset lafan
python preprocess/build_trajectory_inputs.py --dataset 100style --database data/processed/100style_test5/database.npz --tags all --output data/processed/100style_test5/trajectory.npz
```

This writes `trajectory.npz` with:
- `indices`: source pose indices
- `T`: flattened trajectory input `[future_pos, future_dir]`
- `Tpos`: future local positions
- `Tdir`: future local directions

Visualize database + trajectory inputs:

```bash
python Visualization.py --database data/processed/100style_test5/database.npz --trajectory data/processed/100style_test5/trajectory.npz
```

Visualize a full database with the Geno skinned character:

```bash
python Genoview.py --database data/processed/100style_test5/database.npz --trajectory data/processed/100style_test5/trajectory.npz
```

`Genoview.py` requires a full database whose joint count matches the Geno model.
The pruned database is not compatible with Geno skinning.
It uses the project-local `resources/` symlinks to the Geno model and shader files.

If your machine reports desktop OpenGL/GLSL 4.10, patch shader headers once before running:

```bash
python tools/patch_glsl_version.py
```

## Motion Dataset

The encoder-decoder training dataset uses a `ControlOperators`-style motion representation:

- `D_motion = 230` for the pruned skeleton
- split by clip with a fixed `8/1/1` train/val/test ratio
- mirror pairs stay in the same split

Example:

```bash
python -c "from datasets.motion_dataset import MotionDataset; ds = MotionDataset(split='train'); print(ds.split_summary())"
```

## Git

This is a **local-only** repository for project management — it is not pushed
to any remote.

```bash
git status
git add -A
git commit -m "feat: initial commit"
```
