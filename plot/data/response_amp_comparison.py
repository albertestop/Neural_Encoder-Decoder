import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)

sens_data_dir = '/home/albertestop/Sensorium/sensorium/data/sensorium_all_2023/dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20'
lab_data_dir = '/home/albertestop/Sensorium/sensorium/data/sensorium_all_2023/ESPM113'




# Sens Responses
trial = 0
sens_spikes = []
while os.path.exists(os.path.join(sens_data_dir, 'data', 'responses', str(trial) + '.npy')):
    sens_response = np.load(os.path.join(sens_data_dir, 'data', 'responses', str(trial) + '.npy'))
    trial = trial + 1
    for i in range(len(sens_response)):
        peak_indices = find_peaks(sens_response[i, :])[0]
        for j in range(len(peak_indices)):
            sens_spikes.append(sens_response[i, peak_indices[j]])
sens_spikes = np.array(sens_spikes)


# Lab Responses
trial = 0
lab_spikes = []
while os.path.exists(os.path.join(lab_data_dir, 'data', 'responses', str(trial) + '.npy')):
    lab_response = np.load(os.path.join(lab_data_dir, 'data', 'responses', str(trial) + '.npy'))
    trial = trial + 1
    for i in range(len(lab_response)):
        peak_indices = find_peaks(lab_response[i, :])[0]
        for j in range(len(peak_indices)):
            lab_spikes.append(lab_response[i, peak_indices[j]])
lab_spikes = np.array(lab_spikes)

threshold = np.percentile(lab_spikes, 10)
sens_spikes = sens_spikes[sens_spikes >= threshold]
threshold = np.percentile(sens_spikes, 95)
sens_spikes = sens_spikes[sens_spikes <= threshold]
plt.hist(sens_spikes, bins=50, edgecolor='black')
plt.xlabel('Spike amplitude')
plt.ylabel('Frequency')
plt.xlim(0, 80)
plt.title('Sensorium response amplitude')
plt.savefig('plot/data/data_study/Response_amp_sens.png')
plt.cla()
threshold = np.percentile(lab_spikes, 10)
lab_spikes = lab_spikes[lab_spikes >= threshold]
threshold = np.percentile(lab_spikes, 95)
lab_spikes = lab_spikes[lab_spikes <= threshold]
plt.hist(lab_spikes, bins=50, edgecolor='black')
plt.xlabel('Spike amplitude')
plt.ylabel('Frequency')
plt.xlim(0, 200)
plt.title('Lab response amplitude')
plt.savefig('plot/data/data_study/Response_amp_lab.png')