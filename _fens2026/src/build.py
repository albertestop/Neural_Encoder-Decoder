import os
import sys
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import importlib
import pandas as pd
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent.parent
sys.path.append(str(parent_dir))

from _fens2026.src.reconstruct import *

def build_recons(recons_path, mask, time_recons):
    """
    Concatenates a whole reconstruction from the differents segments on the reconstruction directory
    It cuts the reconstruction on the limits of the mask
    """
    save_path = recons_path.parent / Path('whole_session_recons')
    os.makedirs(save_path, exist_ok=True)

    segments = [name for name in os.listdir(recons_path)
        if os.path.isdir(os.path.join(recons_path, name))]
    segments = [s for s in segments if str(s).isdigit()]
    segments = sorted(segments, key=int)
    segments = segments[:-1]

    recons_config_path = str(recons_path) + '/' + segments[0] + '/config.py'
    spec = importlib.util.spec_from_file_location("rec_config", recons_config_path)
    rec_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec_config)

    cols_with_large_values = np.any(mask > rec_config.mask_eval_th, axis=0)
    mask_x_min = np.where(cols_with_large_values)[0][0]
    mask_x_max = np.where(cols_with_large_values)[0][-1]
    rows_with_large_values = np.any(mask > rec_config.mask_eval_th, axis=1)
    mask_y_min = np.where(rows_with_large_values)[0][0]
    mask_y_max = np.where(rows_with_large_values)[0][-1]

    reconstruction = []

    for segment in segments:
        video = np.load(str(recons_path) + '/' + segment + '/video_array.npy')
        reconstruction.append(video[30:-30, :, :])


    video = np.array(reconstruction)        # shape will be (10, 30, 4, 4)
    video = video.reshape(-1, 720, 640)
    video = video[:, (mask_y_min * 10) + 360:((mask_y_max + 1) * 10) + 360, (mask_x_min * 10): ((mask_x_max + 1) * 10)]

    np.save(str(save_path) + '/video_timeline.npy', time_recons)
    np.save(str(save_path) + '/video_array.npy', video)

    print(video.shape)
    iio.imwrite(
        str(save_path) + '/session_recons.mp4',
        video.astype(np.uint8),
        fps=30,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    )

def build_movie(proc_config, rec_config, mask, recons_time, recons_path, t_0, t_f):
    """
    Concatenates each trial video and its corresponding time frames
    In an array with the same timeline as the session reconstructions,
    it moves each frame to the position in the new array whose time stamp
    is closest to the video frame.
    So you get a video frame to its corresponding position.
    """

    trials_df = pd.read_csv('/home/pmateosaparicio/data/Repository/' + proc_config.animal + '/' + proc_config.session + '/' + proc_config.session + '_all_trials.csv')
    trials_df = trials_df[['time', 'duration', 'F1_name']]
    trials_df['F1_name'] = trial_names(trials_df)

    a = trials_df['time'].to_numpy()

    i_start = a.searchsorted(t_0, side="left") - 1
    i_start = max(i_start, 0)
    i_end   = a.searchsorted(t_f, side="right")

    trials_df = trials_df.iloc[i_start:i_end+1].copy()

    time_proj = np.empty((0))
    projections = np.empty((36, 64, 0))

    for name, duration, t_start in zip(trials_df['F1_name'], trials_df['duration'], trials_df['time']):
        trial_video = load_video(name, duration, proc_config.videos_params['videos_dir'])
        projections = np.concatenate((projections, trial_video), axis=2)
        time_proj = np.concatenate((time_proj, np.linspace(float(t_start), float(t_start) + 30, 900)))

    projections = np.transpose(projections, (2, 0, 1))
    mask[mask > rec_config.mask_eval_th] = 1.1
    mask[mask < rec_config.mask_eval_th] = 0.4
    projections = projections * mask
    projections = np.clip(projections, 0, 255)

    projections = projections.repeat(10, axis=1).repeat(10, axis=2)

    proj_video = np.zeros((len(recons_time), 360, 640))

    for i in range(len(recons_time)):
        t = recons_time[i]
        idx = np.abs(time_proj - t).argmin()
        if np.abs(time_proj[idx] - t) < 0.4:
            proj_video[i, :, :] = projections[idx, :, :]

    save_path = recons_path.parent / Path('whole_session_recons')
    np.save(str(save_path) + '/projections_array.npy', proj_video)

    iio.imwrite(
        str(save_path) + '/session_projections.mp4',
        proj_video.astype(np.uint8),
        fps=30,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    )

