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

remote_dir = '/gpfs/projects/uab103/uab020077/Sensorium/scripts'
cp_folds1 = ["ssh", "uab020077@alogin1.bsc.es", f'cd "/gpfs/projects/uab103/uab020077/Sensorium/scripts" && ls']

try:
    subprocess.run(cp_folds1, capture_output=True, text=True, check=True)
    print(subprocess.run(cp_folds1, capture_output=True, text=True, check=True).stdout)

except subprocess.CalledProcessError as e:
    print("SCP failed:", e.returncode)
    print("STDOUT:\n", e.stdout)
    print("STDERR:\n", e.stderr)