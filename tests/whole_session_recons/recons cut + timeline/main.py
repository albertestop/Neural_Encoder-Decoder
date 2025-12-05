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
parent_dir = current_dir.parent.parent.parent
sys.path.append(str(parent_dir))
from src.predictors import generate_predictors
from src.data import get_folds_tiers
from src.responsiveness import responsiveness
import random
import pickle
import imageio.v3 as iio


run_path = '/home/albertestop/Sensorium/Clopath/reconstructions/results/172/2025-07-04_06_ESPM154_004'
segments = [name for name in os.listdir(run_path)
    if os.path.isdir(os.path.join(run_path, name))]
segments = sorted(segments, key=int)
segments = segments[:-1]

reconstruction = []

for segment in segments:
    video = np.load(run_path + '/' + segment + '/video_array.npy')
    reconstruction.append(video[30:-30, :, :])

video = np.array(reconstruction)        # shape will be (10, 30, 4, 4)
video = video.reshape(-1, 720, 640)

video = video[:, 360:515, 330:630]
timeline = np.arange(0, 27600) / 30

np.save(run_path + '/video_timeline.npy', timeline)
np.save(run_path + '/video_array.npy', video)

iio.imwrite(
    run_path + '/session_recons.mp4',
    video.astype(np.uint8),
    fps=30,
    codec="libx264",
    ffmpeg_params=["-pix_fmt", "yuv420p"]
)