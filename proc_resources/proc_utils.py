import numpy as np

def resample(data, original_freq, new_freq):
    original_freq = 20

    duration = len(data) / original_freq
    original_time = np.linspace(0, duration, len(data), endpoint=False)
    new_time = np.linspace(0, duration, int(len(data) * new_freq / original_freq), endpoint=False)

    resampled_data = np.interp(new_time, original_time, data)
    return resampled_data


def split_trial(len_trial, n_frames=300):
    """
    Separate session trial into 300 frame trials
    """

    num_windows = int(len_trial/n_frames) + 1
    num_overlapping_points = num_windows - 1
    reminder_instances = len_trial%n_frames
    overlap = int((n_frames - reminder_instances) / num_overlapping_points) + 1
    window_indices = []
    if overlap*2 < n_frames/3:
        for i in range(num_windows):
            indices_i = i*n_frames - i*overlap
            indices_f = (i + 1)*n_frames - i*overlap
            window_indices.append((indices_i, indices_f))
    else: 
        num_windows = num_windows - 1
        for i in range(num_windows):
            indices_i = i*n_frames
            indices_f = (i + 1)*n_frames
            window_indices.append((indices_i, indices_f))
    window_indices = np.array(window_indices)

    return num_windows, window_indices


def trial_names(trials_df):
    """
    If the F1_name in the experiment_all_trials.csv does not end in a number,
    we assume that the experiment uses the same video in all the experiments,
    defined in config.video_params.videos_dir()
    """

    if trials_df['F1_name'].iloc[1][-1].isdigit():
        trials_df['F1_name'] = trials_df['F1_name'].str[-5:]
    else: 
        trials_df['F1_name'] = ''
    
    return trials_df['F1_name']
