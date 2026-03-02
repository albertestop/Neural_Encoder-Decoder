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

from _fens2026.src.data_loading import *


metrics_session = '2025-07-04_04_ESPM154_008_recons'
run = '0'
train_session = '2025-07-04_04_ESPM154_008'

session_path = "/home/albertestop/data/processed_data/sensorium_all_2023/" + metrics_session
proc_config_path = session_path + "/config.py"
proc_config = load_config(proc_config_path)
recons_path = Path(proc_config.exp_directory + proc_config.animal + '/' + proc_config.session + '/reconstructions/' + run + '/' + metrics_session + '/reconstruction/')
save_path = recons_path.parent / Path('metrics')
random_all_metrics_path = Path(
    "/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_04_ESPM154/"
    "reconstructions/1/2025-07-04_04_ESPM154_008_recons_random_all/metrics"
)
random_time_metrics_path = Path(
    "/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_04_ESPM154/"
    "reconstructions/2/2025-07-04_04_ESPM154_008_recons_random_time/metrics"
)
random_neurons_metrics_path = Path(
    "/home/pmateosaparicio/data/Repository/ESPM154/2025-07-04_04_ESPM154/"
    "reconstructions/3/2025-07-04_04_ESPM154_008_recons_random_neurons/metrics"
)
plot_save_path = Path(
    proc_config.exp_directory
    + proc_config.animal
    + '/'
    + proc_config.session
    + '/reconstructions/'
    + 'plots'
)
os.makedirs(plot_save_path, exist_ok=True)

temporal_corr_evo = np.load(str(save_path) + '/temporal_corr.npy')
temporal_ssim_evo = np.load(str(save_path) + '/temporal_ssim.npy')
spectral_slope_evo = np.load(str(save_path) + '/spectral_slope.npy')
compression_gain_evo = np.load(str(save_path) + '/compression_gain.npy')
random_all_temporal_corr_evo = np.load(random_all_metrics_path / 'temporal_corr.npy')
random_all_temporal_ssim_evo = np.load(random_all_metrics_path / 'temporal_ssim.npy')
random_all_spectral_slope_evo = np.load(random_all_metrics_path / 'spectral_slope.npy')
random_all_compression_gain_evo = np.load(random_all_metrics_path / 'compression_gain.npy')
random_time_temporal_corr_evo = np.load(random_time_metrics_path / 'temporal_corr.npy')
random_time_temporal_ssim_evo = np.load(random_time_metrics_path / 'temporal_ssim.npy')
random_time_spectral_slope_evo = np.load(random_time_metrics_path / 'spectral_slope.npy')
random_time_compression_gain_evo = np.load(random_time_metrics_path / 'compression_gain.npy')
random_neurons_temporal_corr_evo = np.load(random_neurons_metrics_path / 'temporal_corr.npy')
random_neurons_temporal_ssim_evo = np.load(random_neurons_metrics_path / 'temporal_ssim.npy')
random_neurons_spectral_slope_evo = np.load(random_neurons_metrics_path / 'spectral_slope.npy')
random_neurons_compression_gain_evo = np.load(random_neurons_metrics_path / 'compression_gain.npy')
timeline = temporal_corr_evo[:, 0]
skipped_trials_path = '/home/albertestop/data/processed_data/sensorium_all_2023/' + train_session + '/run_data.txt'

with open(skipped_trials_path, "r") as f:
    lines = [line for line in f.readlines() if line.strip()]
skipped_trials = ast.literal_eval(lines[0].split(": ", 1)[1].strip())
skipped_indexes = ast.literal_eval(lines[1].split(": ", 1)[1].strip())

trials_df = pd.read_csv(os.path.join(proc_config.data['session_dir'], proc_config.data['session'] + '_all_trials.csv'))
trials_df = trials_df[['time', 'duration', 'F1_name']]
trials_df['F1_name'] = trials_df['F1_name'].str[-5:] + '/'

seen_trials_df = trials_df.drop(index=skipped_indexes).reset_index(drop=True)
not_seen_trials_df = trials_df.loc[skipped_indexes].reset_index(drop=True)

seen_trials_t = []
inter_trial_t = []
not_seen_trials_t = []
for time, duration in zip(seen_trials_df['time'], seen_trials_df['duration']):
    seen_trials_t.append([time, time + duration])
    inter_trial_t.append([time + duration, time + duration + 5])
for time, duration in zip(not_seen_trials_df['time'], not_seen_trials_df['duration']):
    not_seen_trials_t.append([time, time + duration])
    inter_trial_t.append([time + duration, time + duration + 5])

seen_trials_t = np.array(seen_trials_t)
inter_trial_t = np.array(inter_trial_t)
not_seen_trials_t = np.array(not_seen_trials_t)

seen_temporal_corr_evo = np.empty(0)
seen_temporal_ssim_evo = np.empty(0)
seen_spectral_slope_evo = np.empty(0)
seen_compression_gain_evo = np.empty(0)
for t_i, t_f in seen_trials_t:
    mask = (timeline >= t_i) & (timeline <= t_f)
    seen_temporal_corr_evo = np.concatenate((seen_temporal_corr_evo,(temporal_corr_evo[mask, 1])))
    seen_temporal_ssim_evo = np.concatenate((seen_temporal_ssim_evo,(temporal_ssim_evo[mask, 1])))
    seen_spectral_slope_evo = np.concatenate((seen_spectral_slope_evo,(spectral_slope_evo[mask, 1])))
    seen_compression_gain_evo = np.concatenate((seen_compression_gain_evo,(compression_gain_evo[mask, 1])))

not_seen_temporal_corr_evo = np.empty(0)
not_seen_temporal_ssim_evo = np.empty(0)
not_seen_spectral_slope_evo = np.empty(0)
not_seen_compression_gain_evo = np.empty(0)
for t_i, t_f in not_seen_trials_t:
    mask = (timeline >= t_i) & (timeline <= t_f)
    not_seen_temporal_corr_evo = np.concatenate((not_seen_temporal_corr_evo,(temporal_corr_evo[mask, 1])))
    not_seen_temporal_ssim_evo = np.concatenate((not_seen_temporal_ssim_evo,(temporal_ssim_evo[mask, 1])))
    not_seen_spectral_slope_evo = np.concatenate((not_seen_spectral_slope_evo,(spectral_slope_evo[mask, 1])))
    not_seen_compression_gain_evo = np.concatenate((not_seen_compression_gain_evo,(compression_gain_evo[mask, 1])))

inter_temporal_corr_evo = np.empty(0)
inter_temporal_ssim_evo = np.empty(0)
inter_spectral_slope_evo = np.empty(0)
inter_compression_gain_evo = np.empty(0)
for t_i, t_f in inter_trial_t:
    mask = (timeline >= t_i) & (timeline <= t_f)
    inter_temporal_corr_evo = np.concatenate((inter_temporal_corr_evo,(temporal_corr_evo[mask, 1])))
    inter_temporal_ssim_evo = np.concatenate((inter_temporal_ssim_evo,(temporal_ssim_evo[mask, 1])))
    inter_spectral_slope_evo = np.concatenate((inter_spectral_slope_evo,(spectral_slope_evo[mask, 1])))
    inter_compression_gain_evo = np.concatenate((inter_compression_gain_evo,(compression_gain_evo[mask, 1])))


categories = [
    "Seen trials",
    "Not seen trials",
    "Inter trial",
    "Random all",
    "Random time",
    "Random neurons",
]

def plot_violin(data, categories, title, filename):
    fig, ax = plt.subplots()
    vp = ax.violinplot(data, showmeans=True, showextrema=False)
    for body in vp['bodies']:
        body.set_facecolor('#4C78A8')
        body.set_edgecolor('black')
        body.set_alpha(0.6)

    ax.set_xticks(np.arange(1, len(categories) + 1))
    ax.set_xticklabels(categories)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(plot_save_path / filename, dpi=150)
    plt.close(fig)


plot_violin(
    [
        seen_compression_gain_evo,
        not_seen_compression_gain_evo,
        inter_compression_gain_evo,
        random_all_compression_gain_evo[:, 1],
        random_time_compression_gain_evo[:, 1],
        random_neurons_compression_gain_evo[:, 1],
    ],
    categories,
    "Compression Gain",
    "comp_gain.png",
)

plot_violin(
    [
        seen_spectral_slope_evo,
        not_seen_spectral_slope_evo,
        inter_spectral_slope_evo,
        random_all_spectral_slope_evo[:, 1],
        random_time_spectral_slope_evo[:, 1],
        random_neurons_spectral_slope_evo[:, 1],
    ],
    categories,
    "Spectral Slope",
    "spec_slope.png",
)

plot_violin(
    [
        seen_temporal_corr_evo,
        not_seen_temporal_corr_evo,
        inter_temporal_corr_evo,
        random_all_temporal_corr_evo[:, 1],
        random_time_temporal_corr_evo[:, 1],
        random_neurons_temporal_corr_evo[:, 1],
    ],
    categories,
    "Temporal Correlation",
    "temp_corr.png",
)

plot_violin(
    [
        seen_temporal_ssim_evo,
        not_seen_temporal_ssim_evo,
        inter_temporal_ssim_evo,
        random_all_temporal_ssim_evo[:, 1],
        random_time_temporal_ssim_evo[:, 1],
        random_neurons_temporal_ssim_evo[:, 1],
    ],
    categories,
    "Temporal SSIM",
    "temp_ssim.png",
)
