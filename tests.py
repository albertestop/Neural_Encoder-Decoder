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



recons = np.load('reconstruction_array.npy')[0, 0, 0]
video = np.load('video_array.npy')[0]
summary = np.load('reconstruction_array.npy')[0, 0, 0]

print(recons.shape, video.shape, summary.shape)

plt.imshow(recons, cmap='gray')
plt.axis('off')
plt.savefig('recons.png')

plt.imshow(video, cmap='gray')
plt.axis('off')
plt.savefig('video.png')

plt.imshow(summary, cmap='gray')
plt.axis('off')
plt.savefig('summary.png')