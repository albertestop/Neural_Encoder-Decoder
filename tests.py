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
from PIL import Image

import subprocess
import pandas as pd

data = np.load('/home/albertestop/data/processed_data/sensorium_all_2023/2025-07-04_04_ESPM154_ws_000/data/responses.npy')
timeline = np.load('/home/albertestop/data/processed_data/sensorium_all_2023/2025-07-04_04_ESPM154_ws_000/data/responses_timeline.npy')
print(data.shape, timeline.shape)