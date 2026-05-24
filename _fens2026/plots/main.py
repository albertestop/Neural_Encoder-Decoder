from pathlib import Path
import numpy as np
import ast
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from _fens2026.plots.src import *

"""
    - session: session whose data we will use to do the plots
    - runs: corresponding reconstruction run number inside 
        session_path/reconstructions/run_n corresponding to the session we want to use
    - session_types:
        - movie: Session divided in seen, not_seen, inter for plots
        - sleep: Session divided in active_wake, active_quiet, nrem, rem for plots
        - pupil: Session divided according to corresponding puil diameter for plots
        - random: All session data classified as the type of random it corresponds to

"""

sessions = ['2025-07-04_04_ESPM154_008_recons', '2025-07-04_06_ESPM154_007_sleep', '2025-07-04_04_ESPM154_008_recons_random_all', '2025-07-04_04_ESPM154_008_recons_random_time', '2025-07-04_04_ESPM154_008_recons_random_neurons']#, '2025-07-04_04_ESPM154_008_recons_random_all', '2025-07-04_04_ESPM154_008_recons_random_time', '2025-07-04_04_ESPM154_008_recons_random_neurons']
train_session = '2025-07-04_04_ESPM154_008'
runs = ['0', '0', '1', '2', '3']#, '1', '2', '3']
session_types = ['movie', 'pupil', 'random', 'random', 'random']#, 'random', 'random', 'random']

df = pd.DataFrame(columns=["Temp_Corr", "Temp_SSIM", "Spectral_Slope", "Comp_Gain", "Category"])

for session, run, session_type in zip(sessions, runs, session_types):
    print(f'Preparing session {session}, type = {session_type}')
    session_path = "/home/albertestop/data/processed_data/sensorium_all_2023/" + session
    proc_config_path = session_path + "/config.py"
    proc_config = load_config(proc_config_path)
    preproc_path = Path(proc_config.exp_directory + proc_config.animal + '/' + proc_config.session)
    recons_path = preproc_path / Path('reconstructions/' + run + '/' + session + '/reconstruction/')
    recons_metric_path = recons_path.parent / Path('metrics')
    session_random = proc_config.data['randomize']
    session_random_type = 'random_' + proc_config.data['rand_type']

    temporal_corr_evo = np.load(str(recons_metric_path) + '/temporal_corr.npy')
    temporal_ssim_evo = np.load(str(recons_metric_path) + '/temporal_ssim.npy')
    spectral_slope_evo = np.load(str(recons_metric_path) + '/spectral_slope.npy')
    compression_gain_evo = np.load(str(recons_metric_path) + '/compression_gain.npy')

    timeline = temporal_corr_evo[:, 0]

    if session_type == 'movie':
        trials_df = pd.read_csv(os.path.join(proc_config.data['session_dir'], proc_config.data['session'] + '_all_trials.csv'))
        trials_df = trials_df[['time', 'duration', 'F1_name']]
        trials_df['F1_name'] = trials_df['F1_name'].str[-5:] + '/'

        skipped_trials_path = '/home/albertestop/data/processed_data/sensorium_all_2023/' + train_session + '/run_data.txt'
        with open(skipped_trials_path, "r") as f:
            lines = [line for line in f.readlines() if line.strip()]
        skipped_trials = ast.literal_eval(lines[0].split(": ", 1)[1].strip())
        skipped_indexes = ast.literal_eval(lines[1].split(": ", 1)[1].strip())

        categories = ['seen', 'inter', 'not_seen']
        categories_t = get_movie_categories_t(trials_df, skipped_indexes)
        tot_temporal_corr_evo, tot_temporal_ssim_evo, tot_spectral_slope_evo, tot_compression_gain_evo = segment_movie_session(
            categories_t, timeline, 
            temporal_corr_evo, temporal_ssim_evo, 
            spectral_slope_evo, compression_gain_evo)
        rows = []
        for i in range(len(categories)):
            new_col = np.full((len(tot_temporal_corr_evo[i]), 1), categories[i])
            metrics = np.column_stack((tot_temporal_corr_evo[i], tot_temporal_ssim_evo[i], tot_spectral_slope_evo[i], tot_compression_gain_evo[i], new_col))
            rows.append(pd.DataFrame(metrics, columns=df.columns))
        df_session = pd.concat(rows, ignore_index=True)

    elif session_type == 'sleep':
        categories = ["active_wake", "quiet_wake", "nrem", "rem"]
        categories_t = get_sleep_categories_t(preproc_path)
        tot_temporal_corr_evo, tot_temporal_ssim_evo, tot_spectral_slope_evo, tot_compression_gain_evo = segment_sleep_session(
            categories_t, timeline, 
            temporal_corr_evo, temporal_ssim_evo, 
            spectral_slope_evo, compression_gain_evo)
        rows = []
        for i in range(len(categories)):
            new_col = np.full((len(tot_temporal_corr_evo[i]), 1), categories[i])
            metrics = np.column_stack((tot_temporal_corr_evo[i], tot_temporal_ssim_evo[i], tot_spectral_slope_evo[i], tot_compression_gain_evo[i], new_col))
            rows.append(pd.DataFrame(metrics, columns=df.columns))
        df_session = pd.concat(rows, ignore_index=True)

    elif session_type == 'pupil':
        categories = ["0_10%", "10-30%", "30-70%", "70_100%"]
        categories_t = get_pupil_categories_t(proc_config)
        tot_temporal_corr_evo, tot_temporal_ssim_evo, tot_spectral_slope_evo, tot_compression_gain_evo = segment_sleep_session(
            categories_t, timeline, 
            temporal_corr_evo, temporal_ssim_evo, 
            spectral_slope_evo, compression_gain_evo)
        rows = []
        for i in range(len(categories)):
            new_col = np.full((len(tot_temporal_corr_evo[i]), 1), categories[i])
            metrics = np.column_stack((tot_temporal_corr_evo[i], tot_temporal_ssim_evo[i], tot_spectral_slope_evo[i], tot_compression_gain_evo[i], new_col))
            rows.append(pd.DataFrame(metrics, columns=df.columns))
        df_session = pd.concat(rows, ignore_index=True)
    
    elif session_type == 'random':
        new_col = np.full((len(temporal_corr_evo), 1), session_random_type)
        metrics = np.column_stack((temporal_corr_evo[:, 1], temporal_ssim_evo[:, 1], spectral_slope_evo[:, 1], compression_gain_evo[:, 1], new_col))
        rows.append(pd.DataFrame(metrics, columns=df.columns))
        df_session = pd.concat(rows, ignore_index=True)

    df = pd.concat([df, df_session])

for column in df.columns[:-1]:
    df[column] = df[column].astype(float)
df["Category"] = df["Category"].astype(str)

save_path = Path(os.path.join(proc_config.exp_directory + proc_config.animal, proc_config.session, 'reconstructions', 'plots'))
os.makedirs(save_path, exist_ok=True)

for metric in df.columns[:-1]:
    gen_violin_plot(df, save_path, metric)

gen_dim_reduction_plot(df, save_path)
gen_dim_reduction_min_var_plot(df, save_path)
gen_dim_reduction_3d_plot(df, save_path)
gen_dim_reduction_3d_plot_min_var(df, save_path)