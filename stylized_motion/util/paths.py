"""Stable repository paths used by package modules and command entry points."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = DATA_DIR / "configs"
RESOURCE_DIR = DATA_DIR / "assets" / "genoview"
SOMA_RESOURCE_DIR = DATA_DIR / "assets" / "somaview"
