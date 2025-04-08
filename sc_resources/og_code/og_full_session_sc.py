import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd



def nan_check(eye, pupil_dilation, pupil_pos):
        nan_dilation = np.sum(np.isnan(pupil_dilation))
        nan_pos = np.sum(np.isnan(pupil_pos))
        Prop_nan_dilation = nan_dilation/len(pupil_dilation)
        Prop_nan_pos = nan_pos/len(pupil_pos[0])/2
        print('Nan values for ' + eye + 'eye:\n')
        print('- Pupil dilation: %i (%.2f %%)'%(nan_dilation, Prop_nan_dilation * 100))
        print('- Pupil position: %i (%.2f %%)'%(nan_pos, Prop_nan_pos * 100))
        print(' ')

        return Prop_nan_dilation, Prop_nan_pos


def detect_abrupt_changes(eye, data):
        diffs = np.abs(np.diff(data))  # Calculate absolute differences
        threshold = 10
        indexes = np.where(diffs > threshold)[0]

        print('- ' + eye + ':%.2f %%'%(100* len(indexes)/len(diffs)))


def detect_spike_zones(data, threshold = 5, window_size = 1000, spike_density = 8):
        diffs = np.abs(np.diff(data))  # Calculate absolute differences
        
        indexes = np.where(diffs > threshold)[0]
        
        spike_series = np.zeros(len(data))
        spike_series[indexes] = 1
        spike_counts = pd.Series(spike_series).rolling(window=window_size, min_periods=1).sum()
        spike_zones = np.where(spike_counts >= spike_density)[0]

        return spike_zones


def get_spike_trial_indices(spike_indexes, pupil_time, trial_timestamps):
        """
        Returns trial indices (starting at 1) that contain at least one spike.

        Parameters:
        - spike_zones: List/array of spike indices (frame positions)
        - pupil_time: Array of timestamps corresponding to each frame
        - trial_timestamps: List of trial start timestamps (trial_timestamps[i]) and stop timestamps (trial_timestamps[i+1])

        Returns:
        - trials_with_spikes: List of trial indices (1-based) that contain spikes.
        """

        # Convert spike indices to timestamps
        spike_times = pupil_time[spike_indexes]

        # Set to store unique trial indices with spikes
        trials_with_spikes = set()

        # Assign each spike to its corresponding trial
        for spike_time in spike_times:
            for i in range(len(trial_timestamps) - 1):
                if trial_timestamps[i] <= spike_time < trial_timestamps[i + 1]:
                    trials_with_spikes.add(i + 1)  # Convert 0-based to 1-based index
                    break  # Stop checking once assigned

        return sorted(list(trials_with_spikes))  # Return sorted trial indices

def mask_creator(likelihoods, qc):
        avg_likelihood = np.mean(likelihoods, axis=1)
        mask = 1 - avg_likelihood + qc
        return mask


def response_sanity(data, indexes, pupil_indx, speed_indx, pupil_time, time, resp, hyp, center, dilation, speed_t, speed, excluded, save = False):

    plot_dir = data['plot_dir']
    RorL = data['stim_eye'][0]
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)


        
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(14, 10), layout = 'constrained',
                                    sharex=True, gridspec_kw={'height_ratios': [6, 1, 2, 2, 2, 2]})
    fig.suptitle('Time representation', fontsize=16)
    pcm = ax1.imshow(resp[:, :], cmap='viridis', aspect='auto',
                    extent=[time[0], time[-1], 0, resp.shape[0]], vmin = 0, vmax=0.8)

    ax1.set_yticks([])

    ax2.step(time, hyp)
    ax2.set_yticks([1, 2, 3], ["qa", "nrem", "rem"])
    ax2.grid()
    fig.colorbar(pcm, ax=ax1)
    ax3.plot(pupil_time, center[0, :], zorder = 1)
    ax3.scatter(pupil_time, excluded[0], color='red', s = 2, zorder = 2)
    
    ax4.plot(pupil_time, center[1, :], zorder = 1)
    ax4.scatter(pupil_time, excluded[1], color='red', s = 2, zorder = 2)

    ax5.plot(pupil_time, dilation, zorder = 1)
    ax5.scatter(pupil_time, excluded[2], color='red', s = 2, zorder = 2)


    ax6.plot(speed_t, speed)

    ax3.set_ylabel(RorL + 'eye x')
    ax4.set_ylabel(RorL + 'eye y')
    ax5.set_ylabel(RorL + 'eye z-dilation')
    ax6.set_ylabel('Speed')
    if save:
        plt.savefig(os.path.join(plot_dir, 'Full_recording_sanity'))
    plt.close(fig)



threshold = 0.1
# Getting the data
all_data = [dataleft, dataright]


for i in range(len(all_data)):
    data = all_data[i]
    mouse = data['session']
    eye = data['stim_eye']

    print('==============================================')
    print('Sanity check for ' + data['stim_eye'] + 'eye')
    print('==============================================')

    mouse_dir = constants.processed_dir / mouse
    os.makedirs(os.path.join(mouse_dir, 'data/pupil', eye), exist_ok=True)
    response_rastermap = np.load(os.path.join(mouse_dir, 'data/responses/response_rastermap.npy'))
    response_time = np.load(os.path.join(mouse_dir, 'data/responses/response_time.npy'))
    index_trials = np.load(os.path.join(mouse_dir, 'data/responses/index_trials.npy'))

    index_speed = np.load(os.path.join(mouse_dir, 'data/behavior/index_speed.npy'))

#     pupil_zscored = np.load(os.path.join(mouse_dir, 'data/pupil', eye,  'pupil_zscored.npy'))
    pupil_pos_data = np.load(os.path.join(mouse_dir, 'data/pupil', eye, 'pupil_pos_data.npy'))
    pupil_pos_time = np.load(os.path.join(mouse_dir, 'data/pupil', eye, 'pupil_pos_time.npy'))
    index_pupil = np.load(os.path.join(mouse_dir, 'data/pupil', eye, 'index_pupil.npy'))
    mask = np.load(os.path.join(mouse_dir, 'data/pupil', eye, 'mask_' + eye + '.npy'))

    hypnogram_extended = np.load(os.path.join(mouse_dir, 'data/sleep/hypnogram_extended.npy'))


    speed, speed_time, pupil_dilation, pupil_dilation_time = behavior_proc.load_behavior(data)

    pupil_pos, pupil_pos_time, velocity = pupil_pos_proc.load_pupil_proc(data)
    likelihood_pupil, quality_control = pupil_pos_proc.load_pupil_raw_likelihood(data)

    trials_df = pd.read_csv(os.path.join(data['session_dir'], data['session'] + '_all_trials.csv'))
    trials_df = trials_df[['time', 'duration']]
    trials_times = np.array(trials_df['time'])

    nan_dil_, nan_pos_ = nan_check(data['stim_eye'], pupil_dilation, pupil_pos)

    print('Percentage of abrupt changes for ')

    detect_abrupt_changes(data['stim_eye'] + ' pupil dilation', pupil_dilation)
    detect_abrupt_changes(data['stim_eye'] + ' eye x ', pupil_pos[0])
    detect_abrupt_changes(data['stim_eye'] + ' eye y ', pupil_pos[1])
    print(' ')

    spike_indexes = detect_spike_zones(pupil_dilation)
    print(f"Spike-heavy regions detected at seconds: {pupil_dilation_time[spike_indexes]}")

    spike_trial_indices = get_spike_trial_indices(spike_indexes, pupil_dilation_time, trials_times)
    print("Trials with spikes:", spike_trial_indices)


    plt.figure(figsize=(12, 6))
    plt.xlabel('Frame')
    plt.ylabel('Uncertainty Score') 
    plt.title('DeepLabCut Eye Tracking Confidence Score')
    plt.plot(mask)
    plt.plot([0, len(mask)], [threshold, threshold], ':')

    plt.savefig(os.path.join(mouse_dir, 'data/pupil', eye, 'Sanity_mask'))
    plt.close()

    acceptable_mask = mask < threshold
    excluded_data = np.array([
        np.where(~acceptable_mask, pupil_pos_data[0], np.nan),  # Row 0: pupilX
        np.where(~acceptable_mask, pupil_pos_data[1], np.nan),  # Row 1: pupilY
        np.where(~acceptable_mask, pupil_dilation, np.nan) # Row 2: diameter
    ])
    response_sanity(data, index_trials, index_pupil, index_speed, pupil_pos_time, response_time, response_rastermap,
                              hypnogram_extended, pupil_pos_data, pupil_dilation, speed_time, speed, excluded_data, True)