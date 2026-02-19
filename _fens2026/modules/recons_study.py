from pathlib import Path
import os
import sys

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from _fens2026.src.str_metrics import *
from _fens2026.src.data_loading import *


def compute_recons_metrics(recons_run, metric_window_t):
    window = metric_window_t * 30
    session = read_session('/home/albertestop/Sensorium/Clopath/reconstructions/results/' + recons_run + '/')
    recons_path = Path('/home/albertestop/Sensorium/Clopath/reconstructions/results/' + recons_run + '/' + session)
    recons_config_path = str(recons_path) + '/' + os.listdir(recons_path)[0] + '/config.py'
    rec_config = load_config(recons_config_path)
    proc_config_path = "/home/albertestop/data/processed_data/sensorium_all_2023/" + session + "/config.py"
    proc_config = load_config(proc_config_path)
    mask = load_mask(rec_config, proc_config, session)
    mask = np.where(mask >= rec_config.mask_eval_th, 1, 0)
    mask = mask == 1
    mask = cut_exp_mask(mask, rec_config.mask_eval_th, window)
    recons = np.load(str(recons_path) + '/analysis/whole_session_recons/video_array.npy')
    recons_time = np.load(str(recons_path) + '/analysis/whole_session_recons/video_timeline.npy')

    temporal_corr_evo = np.zeros(recons_time.shape)
    temporal_ssim_evo = np.zeros(recons_time.shape)
    spectral_slope_evo = np.zeros(recons_time.shape)
    compression_gain_evo = np.zeros(recons_time.shape)

    print("Computing metrics")
    for i in range(len(recons_time) - window):
        print(f"{((i/len(recons_time)) * 100):.3f}" + "%")
        window_recons = recons[i:i + window]
        temporal_corr_evo[i + int(window / 2)] = temporal_corr(window_recons, mask)
        temporal_ssim_evo[i + int(window / 2)] = temporal_ssim(window_recons, mask)
        spectral_slope_evo[i + int(window / 2)] = spectral_slope(window_recons, mask)
        compression_gain_evo[i + int(window / 2)] = compression_gain(window_recons, mask)

    save_path = str(recons_path) + '/analysis/metrics'
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + '/temporal_corr.npy', temporal_corr_evo)
    np.save(save_path + '/temporal_ssim.npy', temporal_ssim_evo)
    np.save(save_path + '/spectral_slope.npy', spectral_slope_evo)
    np.save(save_path + '/compression_gain.npy', compression_gain_evo)



if __name__ == '__main__':
    recons_run = '173'
    metric_window_t = 10    # seconds
    compute_recons_metrics(recons_run, metric_window_t)