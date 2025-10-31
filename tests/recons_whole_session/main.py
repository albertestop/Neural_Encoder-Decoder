import sys
import os
from pathlib import Path
import numpy as np
import importlib
import pickle

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
reparent_dir = parent_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(str(reparent_dir))

import imageio.v3 as iio
from Clopath.src.utils_er import *
import pandas as pd
from src import constants

def reconstruct_whole_session(run_path, proc_config):

    print('Reconstructing the whole session')

    segments = [name for name in os.listdir(run_path)
        if os.path.isdir(os.path.join(run_path, name))]
    segments = sorted(segments, key=int)
    print(segments)

    reconstruction = []

    for segment in segments:
        print(segment)
        video = np.load(run_path + '/' + segment + '/video_array.npy')
        reconstruction.append(video[30:-30, :, :])

    print(video.shape)
    video = np.array(reconstruction)
    video = video.reshape(-1, 720, 640)
    video[:, 0:360, :] = 0
    print(video.shape)

    if proc_config.data['s_type'] == 'er':
        session_df = pd.read_csv(os.path.join(proc_config.data['session_dir'], proc_config.data['session'] + '_all_trials.csv'))
        with open(os.path.join(proc_config.data['session_dir'], 'recordings', 's2p_ch0.pickle'), 'rb') as file:
            file = pickle.load(file)
        time = np.array(file['t']).astype(np.float32)
        session_df['t_f'] = session_df['time'] + session_df['duration']
        count = (session_df['t_f'] > (time[-1] - 60)).sum()
        session_df = session_df.iloc[:-count]
        print(session_df)
        pr_videos = []
        for t_0, stim, duration, F1_angle, F1_contrast, F2_angle, F2_contrast in zip(
            session_df['time'], session_df['stim'], session_df['duration'], session_df['F1_angle'], session_df['F1_contrast'], session_df['F2_angle'], session_df['F2_contrast']
            ):
            trial_data = []
            trial_data.append(t_0)
            trial_video = generate_projections(stim, duration, F1_angle, F1_contrast, F2_angle, F2_contrast)
            trial_data.append(trial_video)
            pr_videos.append(trial_data)
        
        for trial in pr_videos:
            t_0 = trial[0]
            tr_video = trial[1]
            t_0_frame = int(t_0 * 30)
            t_f_frame = t_0_frame + tr_video.shape[0]
            video[t_0_frame:t_f_frame,0:360, :] = tr_video

    print('Computing movement direction')

    angles_deg, mean_vecs, speeds = mean_motion_direction_per_frame(video[:, 360:, :])
    mean_vecs[:, 0] = mean_mov(mean_vecs[:, 0], window=15)
    mean_vecs[:, 1] = mean_mov(mean_vecs[:, 1], window=15)

    arrow_video = []
    
    for i in range(video.shape[0]):
        arrow_video.append(arrow_image(x=mean_vecs[i, 0], y=mean_vecs[i, 1], units='coords'))
    arrow_video = np.array(arrow_video)

    video = np.concatenate([video, arrow_video], axis=1)

    iio.imwrite(
        run_path + '/session_recons.mp4',
        video.astype(np.uint8),
        fps=30,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    )

if __name__ == '__main__':
    proc_config_path = "/home/albertestop/data/processed_data/sensorium_all_2023/2025-08-07_05_ESPM163_er2/config.py"
    spec = importlib.util.spec_from_file_location("proc_config", proc_config_path)
    proc_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proc_config)
    session = '/home/albertestop/Sensorium/Clopath/reconstructions/results/157/2025-08-07_05_ESPM163_er2'
    reconstruct_whole_session(session, proc_config)