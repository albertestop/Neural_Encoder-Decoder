import torch
import os
import copy
import numpy as np
from PIL import Image
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd
from src import constants
from scipy.signal import find_peaks


import argus
from argus import load_model
from torch import nn
from src.argus_models import MouseModel
from configs.train_config import config, data_load
from configs.data_proc_001 import data

import numpy as np
import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)
from proc_resources import response_proc, video_proc, behavior_proc, pupil_pos_proc

tiers_lab = np.load(str(constants.sensorium_dir / constants.lab_mice[0] / "meta" / "trials" / "tiers.npy"))
tiers_sens = np.load(str(constants.sensorium_dir / constants.new_mice[0] / "meta" / "trials" / "tiers.npy"))

print(tiers_lab[0])
print(tiers_sens[0])