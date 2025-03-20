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

with open('/home/adamranson/data/Repository/ESMT204/2025-03-05_02_ESMT204/recordings/s2p_ch0.pickle', 'rb') as file:
    file = pickle.load(file)
responses = np.array(file['Spikes']).mean(axis=0)
time = np.array(file['t'])
trial_time = np.array(pd.read_csv('/home/adamranson/data/Repository/ESMT204/2025-03-05_02_ESMT204/2025-03-05_02_ESMT204_all_trials.csv')['time'])

for i in range(30):
    responses = np.load('/home/albertestop/data/processed_data/sensorium_all_2023/2025-03-05_02_ESMT204_000/data/responses/' + str(i) + '.npy')
    print(responses.shape)
