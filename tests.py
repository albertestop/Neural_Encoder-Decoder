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

"""name = 'temporal_ssim.npy'

path = '/home/albertestop/Sensorium/Clopath/reconstructions/results/from_BSC/0/2025-07-04_06_ESPM154_007_sleep/analysis/metrics/' + name
array = np.load(path)
array = array.transpose()
print(array.shape)
os.makedirs('/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_06_ESPM154/reconstruction/metrics', exist_ok=True)
np.save('/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_06_ESPM154/reconstruction/metrics/' + name, array)"""
print('uf')