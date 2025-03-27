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

import shutil
import numpy as np
import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)
from proc_resources import response_proc, video_proc, behavior_proc, pupil_pos_proc

lab_dir = '/home/albertestop/data/processed_data/sensorium_all_2023/2025-02-26_02_ESPM126_001'
old_data_dir = '/home/albertestop/data/processed_data/sensorium_all_2023/dynamic29515'
new_data_dir = '/home/albertestop/data/processed_data/sensorium_all_2023/dynamic29515_2'

tiers = np.load(os.path.join(old_data_dir, 'meta', 'trials', 'tiers.npy'))
tiers = tiers[:360]
np.save(os.path.join(new_data_dir, 'meta', 'trials', 'tiers.npy'), tiers)

# array = np.load(directory_new)
# np.save(directory_new, array[:360])

# files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# for i, file in enumerate(files):
#     file_dir = os.path.join(directory, file)
#     new_file = os.path.join(lab_dir, 'data', 'responses', str(i%250) + '.npy')
#     old_file = np.load(file_dir)
#     array = np.load(new_file)
#     extended_a = np.repeat(array, 5, axis=0)[:array.shape[0], :]

#     np.save(os.path.join(directory_new, file), extended_a)

# for i, file in enumerate(files):
#     file_dir = os.path.join(directory_new, file)
#     print(np.load(file_dir).shape)