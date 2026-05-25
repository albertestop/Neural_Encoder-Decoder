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

sessions = ['2025-07-04_04_ESPM154_008_recons', '2025-07-04_06_ESPM154_007_sleep', '2025-07-04_06_ESPM154_007_sleep', '2025-07-04_06_ESPM154_007_sleep_random_neurons', '2025-07-04_06_ESPM154_007_sleep_random_time', '2025-07-04_06_ESPM154_007_sleep_random_all']#, '2025-07-04_04_ESPM154_008_recons_random_all', '2025-07-04_04_ESPM154_008_recons_random_time', '2025-07-04_04_ESPM154_008_recons_random_neurons']
train_session = '2025-07-04_04_ESPM154_008'
runs = ['0', '0', '0', '1', '2', '3']#, '1', '2', '3']
session_types = ['movie', 'sleep', 'pupil', 'random', 'random', 'random']#, 'random', 'random', 'random']
metrics_used = ["Temp_Corr", "Temp_SSIM", "Spectral_Slope", "Comp_Gain", "Temp_Corr_New", "Temp_Autocorr", "Spectral_Slopoe_New", "Entropy", "Comp_Gain_New", "PCA_Energy", "Frame_Predictab", "Temp_Diff_Energy"]

df = pd.DataFrame(columns=metrics_used + ["Category"])

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

    metrics_raw = []
    metrics_raw.append(np.load(str(recons_metric_path) + '/temporal_corr.npy'))
    metrics_raw.append(np.load(str(recons_metric_path) + '/temporal_ssim.npy'))
    metrics_raw.append(np.load(str(recons_metric_path) + '/spectral_slope.npy'))
    metrics_raw.append(np.load(str(recons_metric_path) + '/compression_gain.npy'))
    metrics_raw.append(np.load(str(recons_metric_path) + '/temporal_corr_new.npy'))
    metrics_raw.append(np.load(str(recons_metric_path) + '/temporal_autocorr.npy'))
    metrics_raw.append(np.load(str(recons_metric_path) + '/spectral_slope_new.npy'))
    metrics_raw.append(np.load(str(recons_metric_path) + '/entropy.npy'))
    metrics_raw.append(np.load(str(recons_metric_path) + '/compression_gain_new.npy'))
    metrics_raw.append(np.load(str(recons_metric_path) + '/pca_energy.npy'))
    metrics_raw.append(np.load(str(recons_metric_path) + '/frame_predictab.npy'))
    metrics_raw.append(np.load(str(recons_metric_path) + '/temp_diff_energy.npy'))

    timeline = metrics_raw[0][:, 0]

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
        total_metrics, total_metrics_t = segment_movie_session(
            categories_t, timeline, 
            metrics_raw)
        rows = []
        for i in range(len(categories)):
            new_col = np.full((len(total_metrics[i][0]), 1), categories[i])
            metrics = np.empty(total_metrics[i][0].shape)
            for cat_metric in total_metrics[i]:
                metrics = np.column_stack((metrics, cat_metric))
            metrics = np.column_stack((metrics, new_col))
            rows.append(pd.DataFrame(metrics[:, 1:], columns=df.columns))
        df_session = pd.concat(rows, ignore_index=True)

    elif session_type == 'sleep':
        categories = ["active_wake", "quiet_wake", "nrem", "rem"]
        categories_t = get_sleep_categories_t(preproc_path)
        total_metrics, total_metrics_t = segment_sleep_session(
            categories_t, timeline, 
            metrics_raw)
        rows = []
        for i in range(len(categories)):
            new_col = np.full((len(total_metrics[i][0]), 1), categories[i])
            metrics = np.empty(total_metrics[i][0].shape)
            for cat_metric in total_metrics[i]:
                metrics = np.column_stack((metrics, cat_metric))
            metrics = np.column_stack((metrics, new_col))
            rows.append(pd.DataFrame(metrics[:, 1:], columns=df.columns))
        df_session = pd.concat(rows, ignore_index=True)

    elif session_type == 'pupil':
        categories = ["0_10%", "10-30%", "30-70%", "70_100%"]
        categories_t = get_pupil_categories_t(proc_config)
        total_metrics, total_metrics_t = segment_sleep_session(
            categories_t, timeline, 
            metrics_raw)
        rows = []
        for i in range(len(categories)):
            new_col = np.full((len(total_metrics[i][0]), 1), categories[i])
            metrics = np.empty(total_metrics[i][0].shape)
            for cat_metric in total_metrics[i]:
                metrics = np.column_stack((metrics, cat_metric))
            metrics = np.column_stack((metrics, new_col))
            rows.append(pd.DataFrame(metrics[:, 1:], columns=df.columns))
        df_session = pd.concat(rows, ignore_index=True)
    
    elif session_type == 'random':
        new_col = np.full((len(metrics_raw[0]), 1), session_random_type)
        metrics = np.empty(metrics_raw[0][:, 1].shape)
        for cat_metric in metrics_raw:
            metrics = np.column_stack((metrics, cat_metric[:, 1]))
        metrics = np.column_stack((metrics, new_col))
        rows.append(pd.DataFrame(metrics[:, 1:], columns=df.columns))
        df_session = pd.concat(rows, ignore_index=True)

    df = pd.concat([df, df_session])
    if session_type != 'random':
        save_path = Path(os.path.join(proc_config.exp_directory + proc_config.animal, proc_config.session, 'reconstructions', 'plots'))
        os.makedirs(save_path, exist_ok=True)
        gen_whole_session_plots(save_path, metrics_used, session_type, categories, timeline, total_metrics_t, metrics_raw)

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