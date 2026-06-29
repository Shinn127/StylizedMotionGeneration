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

Create local links under `data/raw/`:

```bash
mkdir -p data/raw data/processed
ln -s /path/to/lafan data/raw/lafan
ln -s /path/to/100style data/raw/100style
```

If you copied the datasets directly into `data/raw/`, regular directories also work.

Build a database in the style of `ControlOperators-main`:

```bash
python preprocess/build_database.py --dataset lafan
python preprocess/build_database.py --dataset 100style
python preprocess/build_database.py --dataset 100style --max-styles 10 --output data/processed/100style_test10
python preprocess/build_database.py --dataset 100style --max-styles 10 --prune-ends-and-fingers --output data/processed/100style_test10_pruned
```

Both preprocessing entry points support `--workers` for Ubuntu/Linux multi-process acceleration.
On an 8-core desktop, a good starting point is `--workers 8`:

```bash
python preprocess/build_database.py \
  --dataset 100style \
  --max-styles 10 \
  --prune-ends-and-fingers \
  --workers 8 \
  --output data/processed/100style_test10_pruned
```

Outputs are written to `data/processed/<name>/database.npz`.

Build ControlOperators-style trajectory inputs from an existing database:

```bash
python preprocess/build_trajectory_inputs.py --dataset lafan
python preprocess/build_trajectory_inputs.py --dataset 100style --database data/processed/100style_test10/database.npz --tags all --output data/processed/100style_test10/trajectory.npz
python preprocess/build_trajectory_inputs.py --dataset 100style --database data/processed/100style_test10_pruned/database.npz --tags all --output data/processed/100style_test10_pruned/trajectory.npz
```

Trajectory generation also supports `--workers`:

```bash
python preprocess/build_trajectory_inputs.py \
  --dataset 100style \
  --database data/processed/100style_test10_pruned/database.npz \
  --tags all \
  --workers 8 \
  --output data/processed/100style_test10_pruned/trajectory.npz
```

This writes `trajectory.npz` with:
- `indices`: source pose indices
- `T`: flattened trajectory input `[future_pos, future_dir]`
- `Tpos`: future local positions
- `Tdir`: future local directions

Visualize database + trajectory inputs:

```bash
python Visualization.py --database data/processed/100style_test10/database.npz --trajectory data/processed/100style_test10/trajectory.npz
```

Visualize a full database with the Geno skinned character:

```bash
python Genoview.py --database data/processed/100style_test10/database.npz --trajectory data/processed/100style_test10/trajectory.npz
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
- feature packing order:
  - root local linear velocity: `3`
  - root local angular velocity: `3`
  - hips local position: `3`
  - non-root joint rotations in 6D: `24 x 6 = 144`
  - hips local velocity: `3`
  - non-root joint angular velocities: `24 x 3 = 72`
  - foot contacts: `2`
- split by clip with a fixed `8/1/1` train/val/test ratio
- mirror pairs stay in the same split

Example:

```bash
python -c "from datasets.motion_dataset import MotionDataset; ds = MotionDataset(split='train'); print(ds.split_summary())"
```

Feature utilities live in `motion_features.py`:

- `build_motion_features(...)`: pack pruned database motion state into `230D`
- `reconstruct_motion_state_from_features(...)`: recover a visualizable motion state from `230D`

## 230D To Genoview

To validate the `230D -> motion state -> Genoview` reconstruction path, first export a Genoview-compatible database from features:

```bash
cd /Users/shinn/Documents/Projects/StylizedMotionGeneration
conda activate mcc
python features_to_database.py \
  --database data/processed/100style_test5_pruned/database.npz \
  --output data/processed/100style_test5_pruned_roundtrip/database.npz \
  --use-database-motion
cp data/processed/100style_test5_pruned/trajectory.npz \
  data/processed/100style_test5_pruned_roundtrip/trajectory.npz
```

Then visualize the reconstructed database with Geno:

```bash
python Genoview.py \
  --database data/processed/100style_test5_pruned_roundtrip/database.npz \
  --trajectory data/processed/100style_test5_pruned_roundtrip/trajectory.npz
```

To export your own model output features:

```bash
python features_to_database.py \
  --features path/to/your_features.npy \
  --database data/processed/100style_test5_pruned/database.npz \
  --output data/processed/your_recon/database.npz
```

If the feature file stores normalized motion features, add `--normalized`.

## Git

This is a **local-only** repository for project management — it is not pushed
to any remote.

```bash
git status
git add -A
git commit -m "feat: initial commit"
```
