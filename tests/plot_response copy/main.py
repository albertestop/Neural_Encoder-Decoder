import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from src import constants
from configs.data_proc_001 import *
from proc_resources.proc_utils import *

from scipy.signal import find_peaks
session_plot_start = 280
session_plot_end = 380

class Responses_plot:

    def __init__(self, response_params):
        self.params = response_params

    def load_data(self, data):
        """
        It loads the processed two photon recording. It takes the Spike feature of each neuron and creates a 2d array = (neuron, frame).
        It also returns a 1d array corresponding to the time value of the frame.
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
        self.time = self.rec_time + self.rec_period
        self.params = self.params

        self.data = self.spikes
        self.num_neurons = len(self.data[:, 0])


    def process(self, video_frame_time=None):
        """
        Takes the trial data and time from the whole recording by
        using the video frame times of the trial.
        """

        df = pd.DataFrame({
            'responses': self.data[0],
            'time': self.time,
            'point_type': [''] * len(self.data[0])
        })
        exit()

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


    def upscale_data(self, df):
        df.loc[(df['responses'] > df['responses'].shift(1)) & (df['responses'] > df['responses'].shift(-1)), 'point_type'] = 'max'
        print(df["point_type"].value_counts())

        df.loc[(df['responses'] < df['responses'].shift(1)) & (df['responses'] < df['responses'].shift(-1)), 'point_type'] = 'min'
        df.loc[(df['responses'] == df['responses'].shift(1)) & (df['responses'] == df['responses'].shift(-1)), 'point_type'] = 'plain'
        df.loc[(df['responses'] > df['responses'].shift(1)) & (df['responses'] < df['responses'].shift(-1)), 'point_type'] = 'up'
        df.loc[(df['responses'] < df['responses'].shift(1)) & (df['responses'] > df['responses'].shift(-1)), 'point_type'] = 'down'
        df.loc[(df['responses'].shift(1) == 'plain') & (df['responses'] < df['responses'].shift(-1)), 'point_type'] = 'start_up'
        df.loc[(df['responses'].shift(1) == 'plain') & (df['responses'] > df['responses'].shift(-1)), 'point_type'] = 'start_down'
        df.loc[(df['responses'].shift(-1) == 'plain') & (df['responses'] > df['responses'].shift(1)), 'point_type'] = 'end_up'
        df.loc[(df['responses'].shift(-1) == 'plain') & (df['responses'] < df['responses'].shift(1)), 'point_type'] = 'end_down'
        if df.loc[0, 'responses'] < df.loc[1, 'responses']:
            df.loc[0, 'point_type'] = 'start_up'
        elif df.loc[0, 'responses'] > df.loc[1, 'responses']:
            df.loc[0, 'point_type'] = 'start_down'
        else:
            df.loc[0, 'point_type'] = 'plain'
        
        print(df["point_type"].value_counts())
            
        # Generate Upscaled Time 
        upscale_ratio = int(60 / self.rec_freq)
        if upscale_ratio % 2 == 0: upscale_ratio = upscale_ratio + 1
        print(upscale_ratio)
        upscaled_time = np.zeros((upscale_ratio * len(self.time)))
        indexes = np.full(len(upscaled_time), False, dtype=bool)
        for i in range(len(self.time)):
            index = i * upscale_ratio + int(upscale_ratio / 2)
            indexes[index] = True
        upscaled_time[indexes] = self.time
        x = np.arange(len(upscaled_time))
        upscaled_time[~indexes] = np.interp(x[~indexes], x[indexes], self.time)

        
        upscaled_data = np.zeros(upscaled_time.shape)
        # Plain dots
        for i in df.index[df['point_type'] == 'plain']:
            upscale_index = (i * upscale_ratio) + int(upscale_ratio / 2)
            start = upscale_index - int(upscale_ratio / 2)
            end = min(upscale_index + int(upscale_ratio / 2), len(df) - 1)
            upscaled_data[start:end] = df.loc[i, 'responses']
        print('1')
        # Up and down dots
        for i in df.index[(df['point_type'] == 'up') | (df['point_type'] == 'down')]:
            upscale_index = i * upscale_ratio + int(upscale_ratio / 2)
            start = max(upscale_index - int(upscale_ratio / 2), 0)
            end = min(upscale_index + int(upscale_ratio / 2), len(df) - 1)
            upscaled_data[start:end] = np.interp(upscaled_time[start:end], [df.loc[i - 1:i + 1, 'time']], [df.loc[i - 1:i + 1, 'responses']])
        print('2')
        # Max dots
        for i in df.index[df['point_type'] == 'max']:
            upscale_index = i * upscale_ratio + int(upscale_ratio / 2)
            start = max(upscale_index - int(upscale_ratio / 2), 0)
            end = min(upscale_index + int(upscale_ratio / 2), len(df) - 1)
            for j in range(end - start):
                upscaled_data[start + j] = -(4 / (upscale_ratio * upscale_ratio)) * ((j - (upscale_ratio / 2)) * (j - (upscale_ratio / 2))) + df.loc[i, 'responses'] + 0.5
        print('3')
        # Min dots
        for i in df.index[df['point_type'] == 'min']:
            upscale_index = i * upscale_ratio + int(upscale_ratio / 2)
            start = max(upscale_index - int(upscale_ratio / 2), 0)
            end = min(upscale_index + int(upscale_ratio / 2), len(df) - 1)
            for j in range(end - start):
                upscaled_data[start + j] = (4 / (upscale_ratio * upscale_ratio)) * ((j - (upscale_ratio / 2)) * (j - (upscale_ratio / 2))) + df.loc[i, 'responses'] - 0.5
        print('4')
        # End_up, Start_down, End_down dots
        print(len(df.index[(df['point_type'] == 'end_up') | (df['point_type'] == 'start_down') | (df['point_type'] == 'end_down')]))
        for i in df.index[(df['point_type'] == 'end_up') | (df['point_type'] == 'start_down') | (df['point_type'] == 'end_down')]:
            print(i)
            upscale_index = i * upscale_ratio + int(upscale_ratio / 2)
            start = max(upscale_index - int(upscale_ratio / 2), 0)
            end = min(upscale_index + int(upscale_ratio / 2), len(df) - 1)
            upscaled_data[start:end] = np.interp(upscaled_time[start:end] , [upscaled_time[start - 1], upscaled_time[end + 1]], [upscaled_data[start - 1], upscaled_data[end + 1]])
        
        self.upscaled_time = upscaled_time
        self.upscaled_data = upscaled_data
        return 0

        










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


with open(os.path.join(data['session_dir'], 'recordings', 's2p_ch0.pickle'), 'rb') as file:
    file = pickle.load(file)


trials_df = pd.read_csv(os.path.join(data['session_dir'], data['session'] + '_all_trials.csv'))
trials_df = trials_df[['time', 'duration', 'F1_name']]
trials_df['F1_name'] = trial_names(trials_df)

responses = Responses_plot(response_params)
responses.load_data(data)


j = 0
mouse = data['session'] + '_' + data['mouse_run']
mouse_dir = constants.sensorium_dir / mouse

# Print 
print(len(responses.time[(responses.time >= 10) & (responses.time < 11)]))
print(responses.time[session_plot_start+18:session_plot_start+48])
print(responses.data[30, session_plot_start+18:session_plot_start+48])


# PLOT RAW DATA

# Plot spikes

plt.plot(responses.time[session_plot_start:session_plot_end], responses.spikes[30, session_plot_start:session_plot_end], label='Spikes')
plt.ylabel('Response Amplitude')
plt.xlabel('Time')
plt.title('Response time coherence')
plt.legend()
plt.savefig('spikes.png')
plt.cla()


# Plot dF

plt.plot(responses.time[session_plot_start:session_plot_end], responses.dF[30, session_plot_start:session_plot_end], label='dF')
plt.ylabel('Response Amplitude')
plt.xlabel('Time')
plt.title('Response time coherence')
plt.legend()
plt.savefig('dF.png')
plt.cla()


# Plot F

plt.plot(responses.time[session_plot_start:session_plot_end], responses.F[30, session_plot_start:session_plot_end], label='F')
plt.ylabel('Response Amplitude')
plt.xlabel('Time')
plt.title('Response time coherence')
plt.legend()
plt.savefig('F.png')
plt.cla()


# Plot dF/F

plt.plot(responses.time[session_plot_start:session_plot_end], responses.dF_F[30, session_plot_start:session_plot_end], label='dF/F')
plt.ylabel('Response Amplitude')
plt.xlabel('Time')
plt.title('Response time coherence')
plt.legend()
plt.savefig('dF_F.png')
plt.cla()

responses.process()
exit()


# Plot mean data

response_mean = np.mean(responses.data, axis=0)
response_means, time_values = np.array([]), np.array([])
for t_0 in trials_df['time'].iloc[0:]:
    indexes = np.where((responses.time > t_0 - 8) & (responses.time < t_0 + 8))
    response_means = np.concatenate((response_means, response_mean[indexes]))
    time_values = np.concatenate((time_values, responses.time[indexes] - t_0))
df = pd.DataFrame({
    'responses': response_means,
    'time': time_values
})
df.sort_values(by='time', ascending=True, inplace=True)
df['time'] = (df['time'] / 0.05).round() * 0.05
df = df.groupby('time', as_index=False)['responses'].mean()
responses_tot = np.array(df['responses'])
time_tot = np.array(df['time'])
response_mean_amp = np.mean(responses_tot)
rec_rate = len(np.where((responses.time >= 10) & (responses.time < 11))[0])
plt.plot(time_tot, responses_tot, label='response_mean')
plt.axhline(y=response_mean_amp, color='green', linestyle='--', label='mean session response amp')
plt.axvline(x=0, color='red', linestyle='--', label='trial_start')
plt.ylabel('Response Amplitude')
plt.xlabel('Time')
plt.title('Response time coherence')
plt.legend()
plt.savefig('delete.png')
plt.cla()


# PLOT PROCESSED DATA

