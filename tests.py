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

data = np.load('/home/albertestop/data/processed_data/sensorium_all_2023/2025-04-01_01_ESPM127_014_recons/data/responses/4.npy')
plt.clf()
plt.imshow(data, aspect='auto', vmin=0, vmax=data[data != 0].mean())
plt.colorbar(label='Response Intensity')
plt.xlabel('Frame')
plt.ylabel('Neuron')
plt.title('Lab response')
plt.savefig('delete.png')