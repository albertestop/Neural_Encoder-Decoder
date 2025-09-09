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


dataset_id = '2025-06-13_01_ESPM135_006'

import sys
import importlib.util

module_path = "/home/albertestop/data/processed_data/sensorium_all_2023/2025-07-08_04_ESPM152_000/config.py"

spec = importlib.util.spec_from_file_location("d_config", module_path)
d_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d_config)

# now you can use it
print(d_config.animal)
