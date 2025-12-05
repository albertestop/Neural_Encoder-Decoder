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
from PIL import Image

mask = np.load('/home/albertestop/Sensorium/Clopath/reconstructions/masks/mask_2025-07-04_04_ESPM154_004.npy')
trials_df = pd.read_csv('/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_04_ESPM154/2025-07-04_04_ESPM154_all_trials.csv')
videos_dir = '/data/Remote_Repository/bv_resources/all_movie_clips_bv_sets/002'
run_path = '/home/albertestop/Sensorium/Clopath/reconstructions/results/173/2025-07-04_04_ESPM154_004_concat'

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

t_0 = 70 * 10
t_f = t_0 + (len(segments) * 10)
fr_i = t_0 * 30
fr_f = fr_i + (len(segments) * 10 * 30)
time_recons = np.arange(fr_i, fr_f) / 30

np.save(run_path + '/video_timeline.npy', time_recons)
np.save(run_path + '/video_array.npy', video)

print(video.shape)
iio.imwrite(
    run_path + '/session_recons.mp4',
    video.astype(np.uint8),
    fps=30,
    codec="libx264",
    ffmpeg_params=["-pix_fmt", "yuv420p"]
)


def trial_names(trials_df):
    """
    If the F1_name in the experiment_all_trials.csv does not end in a number,
    we assume that the experiment uses the same video in all the experiments,
    defined in config.video_params.videos_dir()
    """

    if trials_df['F1_name'].iloc[1][-1].isdigit():
        trials_df['F1_name'] = trials_df['F1_name'].str[-5:] + '/'
    else: 
        trials_df['F1_name'] = ''
    
    return trials_df['F1_name']


def load_video(video_id, duration, videos_dir):

    video_lab = []
    for i in range(duration * 30):
        frame_id = str((i // 100)%10) + str((i // 10)%10) + str(i%10)
        frame = load_frame(video_id, frame_id, videos_dir)
        frame = frame.resize((64, 36))
        video_lab.append(np.array(frame))

    trial_video = np.transpose(np.array(video_lab).astype(np.float32), (1, 2, 0)).astype(np.float32)
    return trial_video


def load_frame(video_id, frame_id, videos_dir):
    frame_path = video_id + 'frame-' + frame_id + '.jpg'
    frame = Image.open(videos_dir + '/' + frame_path)
    return frame


trials_df = trials_df[['time', 'duration', 'F1_name']]
trials_df['F1_name'] = trial_names(trials_df)

a = trials_df['time'].to_numpy()

i_start = a.searchsorted(t_0, side="left") - 1
i_end   = a.searchsorted(t_f, side="right")

trials_df = trials_df.iloc[i_start:i_end+1].copy()

time_proj = np.empty((0))
projections = np.empty((36, 64, 0))

for name, duration, t_start in zip(trials_df['F1_name'], trials_df['duration'], trials_df['time']):
    trial_video = load_video(name, duration, videos_dir)
    projections = np.concatenate((projections, trial_video), axis=2)
    time_proj = np.concatenate((time_proj, np.arange(float(t_start), float(t_start) + 30, 1/30)))

projections = np.transpose(projections, (2, 0, 1))
mask = mask[14:-14, :]
mask[mask > 0.75] = 1.1
mask[mask < 0.75] = 0.4
projections = projections * mask
projections = np.clip(projections, 0, 255)

projections = projections.repeat(10, axis=1).repeat(10, axis=2)

proj_video = np.empty((len(video), 360, 640))

for i in range(len(video)):
    t = time_recons[i]
    idx = np.abs(time_proj - t).argmin()
    if np.abs(time_proj[idx] - t) < 0.4:
        proj_video[i, :, :] = projections[idx, :, :]


np.save(run_path + '/projections_array.npy', proj_video)
print(proj_video.shape)
iio.imwrite(
    run_path + '/session_projections.mp4',
    proj_video.astype(np.uint8),
    fps=30,
    codec="libx264",
    ffmpeg_params=["-pix_fmt", "yuv420p"]
)