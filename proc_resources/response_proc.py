import pickle
import numpy as np
import os
import pandas as pd

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
        spikes = np.array(file['Spikes']).astype(np.float32)
        dF = np.array(file['dF']).astype(np.float32)
        F = np.array(file['F']).astype(np.float32)
        dF_F = dF/F
        time = np.array(file['t']).astype(np.float32)

        self.n_planes = n_planes
        self.rec_freq = rec_freq
        self.rec_period = rec_period
        self.spikes = spikes
        self.dF = dF
        self.F = F
        self.dF_F = dF_F
        self.rec_time = time
        self.time_global = self.rec_time + (self.rec_period/2)

        self.data = self.spikes
        self.params = self.params
        self.num_neurons = len(self.data[:, 0])


    def compute_neuron_planes(self):
        """
        Array with the plane of each neuron.
        
        It computes the index of the first different value of each neuron
        and applies % n_planes
        """
        mask = self.data != self.data[:, 0][:, None]
        first_diff_idx = np.argmax(mask, axis=1)
        first_diff_idx[~mask.any(axis=1)] = -1
        self.neuron_plane = first_diff_idx % self.n_planes


    def compute_plane_times(self):
        self.time = np.zeros((self.n_planes, int(np.floor(self.time_global.shape[0]/self.n_planes))))
        for i in range(self.n_planes):
            plane_time = self.time_global[i::9]
            if len(self.time[i, :]) < len(self.time_global[i::9]):
                self.time[i, :] = plane_time[:-1]
            else:
                self.time[i, :] = plane_time


    def downscale(self):
        downs_resp = np.zeros((self.data.shape[0], int(np.floor(self.data.shape[1]/self.n_planes))))
        for i in range(len(downs_resp)):
            neuron_data = self.data[i, self.neuron_plane[i]::9]
            if len(downs_resp[i, :]) < len(self.data[i, self.neuron_plane[i]::9]):
                downs_resp[i] = neuron_data[:-1]
            else:
                downs_resp[i] = neuron_data

        self.data = downs_resp


    def upscale(self):
        upscale_ratio = 60 / self.rec_freq
        dx = self.rec_period
        for i in range(len(self.data)):
            first_derivative = np.gradient(self.data[i, :], dx)
            second_derivative = np.gradient(first_derivative, dx)
            point_types = np.sign(second_derivative)
            point_types[self.data[i] == 0] = 0
            window_indexes = np.where(point_types == 1)[0]
            for i in range(len(window_indexes)):
                window_start = window_indexes[i]
                window_end = window_indexes[i + 1]
                window_data = self.data[window_start:window_end]
                window_upscaled = np.array((len(window_data) * upscale_ratio))

        pass


    def keep_only_spikes(self, video_freq):
        temp_time = np.arange(0, self.time_global[-1], 1/video_freq)

        resampled_response = []
        for i in range(len(self.data)):
            resampled_response.append(np.interp(temp_time, self.time[self.neuron_plane[i]], self.data[i, :]).astype(np.float32))
        temp_data = np.array(resampled_response)

        temp_data2 = np.zeros(temp_data.shape)
        for i in range(len(self.data)):
            peak_indices = find_peaks(temp_data[i, :])[0]
            temp_data2[i, peak_indices] = temp_data[i, peak_indices]

        self.data, self.time = temp_data2, temp_time


    def process_global(self, video_freq):
        self.compute_neuron_planes()

        self.compute_plane_times()

        if self.params['downscale']:
            self.downscale()
        
        elif self.params['upscale']:
            self.upscale()

        if self.params['keep_only_spikes']: 
            self.keep_only_spikes(video_freq)


    def resample(self, video_frame_time, time):
        resampled_response = []
        for i in range(self.num_neurons):
            resampled_response.append(np.interp(video_frame_time, time[self.neuron_plane[i]], self.data[i, :]))
        self.trial_data = np.array(resampled_response)


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


    def process(self, video_frame_time):
        """
        Takes the trial data and time from the whole recording by
        using the video frame times of the trial.
        """
        if self.params['resample']:
            self.resample(video_frame_time, self.time)

        if self.params['responses_renorm']:
            self.renorm()
        self.trial_data = self.trial_data.astype(np.float32)