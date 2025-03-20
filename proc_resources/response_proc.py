import pickle
import numpy as np
import os

from scipy.signal import find_peaks

class Responses:

    def __init__(self, response_params):
        self.params = response_params

    def load_data(self, data):
        """
        It loads the processed two photon recording.

        A data input of responses.data[30, i] corresponds to 
        the amount of light recorded from responses.time[30] to 
        responses.time[30] + rec_period. Therefore, it should be delayed by
        rec_period/2
        The recording data we load, before loading it is upscaled to 30 Hz 
        repeating points between data inputs.
        """
        with open(os.path.join(data['session_dir'], 'recordings', 's2p_ch0.pickle'), 'rb') as file:
            file = pickle.load(file)

        n_planes = len(np.unique(np.array(file['Depths'])))
        rec_freq = 30/n_planes
        rec_period = 1/rec_freq
        spikes = np.array(file['Spikes'])
        dF = np.array(file['dF'])
        F = np.array(file['F'])
        dF_F = dF/F
        time = np.array(file['t'])

        self.n_planes = n_planes
        self.rec_freq = rec_freq
        self.rec_period = rec_period
        self.spikes = spikes
        self.dF = dF
        self.F = F
        self.dF_F = dF_F
        self.rec_time = time
        self.time = self.rec_time + (self.rec_period/2)
        self.params = self.params

        self.data = self.spikes
        self.num_neurons = len(self.data[:, 0])


    def process(self, video_frame_time):
        """
        Takes the trial data and time from the whole recording by
        using the video frame times of the trial.
        """
        if self.params['keep_only_spikes']: 
            pre_time = self.keep_only_spikes(video_frame_time)
        else:
            pre_time, self.trial_data = self.time, self.data

        resampled_response = []
        for i in range(self.num_neurons):
            resampled_response.append(np.interp(video_frame_time, pre_time, self.trial_data[i, :]))
        self.trial_data = np.array(resampled_response)

        if self.params['responses_renorm']:
            self.renorm()


    def keep_only_spikes(self, video_frame_time):
        temp_time = video_frame_time[::4]

        resampled_response = []
        for i in range(len(self.data)):
            resampled_response.append(np.interp(temp_time, self.time, self.data[i, :]))
        self.trial_data = np.array(resampled_response)

        self.trial_data = np.zeros(self.trial_data.shape)
        for i in range(len(self.data)):
            peak_indices = find_peaks(self.trial_data[i, :])[0]
            self.trial_data[i, peak_indices] = self.trial_data[i, peak_indices]

        return temp_time


    def renorm(self):
        spikes = []
        for i in range(len(self.trial_data)):
            peak_indices = find_peaks(self.trial_data[i, :])[0]
            for index in peak_indices:
                spikes.append(self.trial_data[i, index])
        spikes = np.array(spikes)
        nonzero_count = np.count_nonzero(spikes)
        
        if nonzero_count > 0:
            threshold = np.percentile(spikes, 10)
            spikes = spikes[spikes >= threshold]
            spike_mean = np.mean(spikes)

            if self.params['renorm'] == 'sens_renorm':
                self.trial_data = 16.97*self.trial_data/spike_mean