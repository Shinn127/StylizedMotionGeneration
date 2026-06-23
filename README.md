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
├── configs/              # training / inference configs
├── data/                 # datasets (git-ignored)
├── models/               # model definitions
├── train.py
├── generate.py
└── utils/
```

## Git

This is a **local-only** repository for project management — it is not pushed
to any remote.

```bash
git status
git add -A
git commit -m "feat: initial commit"
```
