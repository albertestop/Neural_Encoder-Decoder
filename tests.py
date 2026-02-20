import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import json
import torch

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
from src.predictors import generate_predictors
from src.data import get_folds_tiers
from src.responsiveness import responsiveness
import random
import pickle
import imageio.v3 as iio
import subprocess
from src import constants

def swap_recons_and_name(path_str: str) -> Path:
    """
    Given a path whose last folders are .../recons/name/,
    move contents so that the new structure is .../name/recons/.

    Returns the new base path (.../name/recons/).
    """
    path = Path(path_str).resolve()

    # Expect something like .../recons/name[/optional_subpath]
    name_dir = path
    recons_dir = name_dir.parent
    parent = recons_dir.parent

    # New structure: .../name/recons[/optional_subpath]
    new_base = parent / name_dir.name / recons_dir.name

    # If the input path points inside `name`, preserve the subpath
    subpath = path.relative_to(name_dir)
    new_path = new_base / subpath

    # Create new_base if needed
    new_base.mkdir(parents=True, exist_ok=True)
    print(new_base)

    # Move all entries from old name_dir to new_base
    for item in name_dir.iterdir():
        target = new_base / item.name
        shutil.move(str(item), str(target))

    # Optionally remove the now-empty old `name` and `recons` dirs if desired:
    name_dir.rmdir()
    recons_dir.rmdir()

    return new_path

old = "/home/pmateosaparicio/data/Repository/ESPM127/2025-04-01_02_ESPM127/reconstructions/0/reconstruction/2025-04-01_02_ESPM127_013_sleep"
new = swap_recons_and_name(old)