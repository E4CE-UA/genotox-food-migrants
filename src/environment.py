"""Seeds and version logging, so the notebook is reproducible."""

import os
import platform
import random
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

SEED = 20260925
SEEDS = (0, 1, 2, 3, 4)  # seeds for the scaffold splits

PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "rdkit",
    "lightgbm",
    "matplotlib",
    "marimo",
    "pyarrow",
)


def set_seeds(seed: int = SEED) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def versions() -> pd.DataFrame:
    """Version table to keep a record in the first cell."""
    rows = [("python", sys.version.split()[0]), ("platform", platform.platform())]
    rows += [(p, _version(p)) for p in PACKAGES]
    return pd.DataFrame(rows, columns=["component", "version"])
