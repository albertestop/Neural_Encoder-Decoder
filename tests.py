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

array = np.load('/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_04_ESPM154/reconstructions/0/2025-07-04_04_ESPM154_008_recons/whole_session_recons/mask_contour_coords.npy')
array = np.transpose(array)
print(array.shape)
np.save('/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_04_ESPM154/reconstructions/0/2025-07-04_04_ESPM154_008_recons/whole_session_recons/mask_contour_coords.npy', array)