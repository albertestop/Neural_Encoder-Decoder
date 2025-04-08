import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from rastermap import Rastermap, utils
from scipy.stats import zscore
import pickle


def sc_time_coherence(trials_time, responses, behavior, pupil_pos):

    t_i, t_f = np.array(trials_time)[0], np.array(trials_time)[-1]
    if t_i - 8 > responses.time_global[0] and t_i - 8 > behavior.speed_time[0] and t_i - 8 > behavior.pupil_dilation_time[0] and t_i - 8 > pupil_pos.time[0]:
        t_i_check = True 
    else: 
        t_i_check = False
    if t_f + 8 < responses.time_global[-1] and t_f + 8 < behavior.speed_time[-1] and t_f + 8 < behavior.pupil_dilation_time[-1] and t_f + 8 < pupil_pos.time[-1]:
        t_f_check = True
    else: t_f_check = False

    print('Recording start time and end time checks: ' + str(t_i_check) + ', ' + str(t_f_check))

    response_synchro = (len(responses.data[0, :]) == len(responses.time_global))
    speed_synchro = (len(behavior.speed) == len(behavior.speed_time))
    pupil_dilation_synchro = (len(behavior.pupil_dilation) == len(behavior.pupil_dilation_time))
    pupil_pos_synchro = (len(pupil_pos.data[0, :]) == len(pupil_pos.time))

    print('Data synchronized: ' + str((response_synchro and speed_synchro and pupil_dilation_synchro and pupil_pos_synchro)))
    if not response_synchro: print('Response data not synchronyzed. Number of time inputs = ' + str(len(responses.time_global)) + ', number of response inputs = ' + str(len(responses.data[0, :])))
    if not speed_synchro: print('Speed data not synchronyzed. Number of time inputs = ' + str(len(behavior.speed_time)) + ', number of speed inputs = ' + str(len(behavior.speed)))
    if not pupil_dilation_synchro: print('Pupil dilation data not synchronyzed. Number of time inputs = ' + str(len(behavior.pupil_dilation_time)) + ', number of pupil dilation inputs = ' + str(len(behavior.pupil_dilation)))
    if not pupil_pos_synchro: print('Pupil position data not synchronyzed. Number of time inputs = ' + str(len(pupil_pos.time)) + ', number of pupil position inputs = ' + str(len(pupil_pos.data[0, :])))


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


def load_sleep_proc(data):
    with open(os.path.join(data['session_dir'], 'sleep_analysis', 'scoring_datav3.pickle'), 'rb') as file:
        sleep_data = pickle.load(file)

    hypnoraw = sleep_data['hypnogram']
    timesleep = sleep_data['epoch_time']
    return hypnoraw, timesleep


def comp_response_rastermap(response, clusters = 50, upsample = 10, nbin = 1):
    """
    This function takes the response data, computes the optimal number of PCAs, 
    z-scores the responses and then applies rastermap algorithm reordering
    neuron spikes.
    """
    print('Computing rastermap-ization')
    spks = zscore(response)
    optimal_n_PCs = 120

    model = Rastermap(n_clusters=clusters, # number of clusters to compute
                  n_PCs=optimal_n_PCs, # number of PCs to use
                  locality=0., # locality in sorting to find sequences (this is a value from 0-1)
                  grid_upsample=upsample, # default value, 10 is good for large recordings
                ).fit(spks)
    
    isort = model.isort
    # nbin = 1 # number of neurons to bin over 
    sn = utils.bin1d(spks[isort], bin_size=nbin, axis=0) # bin over neuron axis
    return sn


def response_sanity(parent_dir, data, eye, responses, pupil_pos, behavior, resp_raster, excluded, hyp, save = False):

    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(14, 10), layout = 'constrained',
                                    sharex=True, gridspec_kw={'height_ratios': [6, 1, 2, 2, 2, 2]})
    fig.suptitle('Time representation', fontsize=16)
    pcm = ax1.imshow(resp_raster[:, :], cmap='viridis', aspect='auto',
                    extent=[responses.time_global[0], responses.time_global[-1], 0, resp_raster.shape[0]], vmin = 0, vmax=0.8)

    ax1.set_yticks([])

    ax2.step(responses.time_global, hyp)
    ax2.set_yticks([1, 2, 3], ["qa", "nrem", "rem"])
    ax2.grid()
    fig.colorbar(pcm, ax=ax1)
    ax3.plot(pupil_pos.time, pupil_pos.data[0, :], zorder = 1)
    ax3.scatter(pupil_pos.time, excluded[0], color='red', s = 2, zorder = 2)
    
    ax4.plot(pupil_pos.time, pupil_pos.data[1, :], zorder = 1)
    ax4.scatter(pupil_pos.time, excluded[1], color='red', s = 2, zorder = 2)

    ax5.plot(behavior.pupil_dilation_time, behavior.pupil_dilation, zorder = 1)
    ax5.scatter(behavior.pupil_dilation_time, excluded[2], color='red', s = 2, zorder = 2)

    ax6.plot(behavior.speed_time, behavior.speed)

    ax3.set_ylabel(eye + 'eye x')
    ax4.set_ylabel(eye + 'eye y')
    ax5.set_ylabel(eye + 'eye z-dilation')
    ax6.set_ylabel('Speed')
    if save:
        plt.savefig(os.path.join(parent_dir, 'pre_processing', data['session'], 'sanity_check', 'Full_session_sc.png'))
    plt.close(fig)



def full_session_sc(parent_dir, data, responses, behavior, pupil_pos):
    eyes = ['Left', 'Right']

    resp_raster = comp_response_rastermap(responses.data)

    if data['sleep']:
        hypnoraw, timesleep = load_sleep_proc(data)
        indices = np.searchsorted(timesleep, responses.time_global)
        indices = np.clip(indices, 0, len(timesleep) - 1)
        hypnogram_extended = hypnoraw[indices]
    else:
        hypnogram_extended = np.zeros(responses.time_global.shape)

    for i, eye in enumerate(eyes):

        excluded_data = np.array([
            np.where(~pupil_pos.acceptable_mask[i], pupil_pos.data[0], np.nan),  # Row 0: pupilX
            np.where(~pupil_pos.acceptable_mask[i], pupil_pos.data[1], np.nan),  # Row 1: pupilY
            np.where(~behavior.acceptable_mask[i], behavior.pupil_dilation, np.nan) # Row 2: diameter
        ])

        response_sanity(
            parent_dir, data, eye, responses, pupil_pos, behavior, resp_raster, excluded_data, hypnogram_extended, True
        )