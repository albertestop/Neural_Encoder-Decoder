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
from src.data import save_fold_tiers
from configs.train_config import config, data_load
from configs.data_proc_001 import data

import shutil
import numpy as np
import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)
from proc_resources import response_proc, video_proc, behavior_proc, pupil_pos_proc

save_fold_tiers("2025-02-26_02_ESPM126_000")