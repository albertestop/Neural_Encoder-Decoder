import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import pickle

animal = 'ESPM163'
session = '2025-08-07_05_ESPM163'
exp_directory = '/home/melinatimplalexi/data/Repository/'
session_dir = str(os.path.join(exp_directory, animal, session))

with open(os.path.join(session_dir, 'recordings', 's2p_oasis_ch0.pickle'), 'rb') as file:
    file = pickle.load(file)
spikes = np.array(file['oasis_spikes']).astype(np.float32)
time = np.array(file['t']).astype(np.float32)
mean_act = spikes.mean(axis=0)
print(np.isnan(spikes).sum())
print(np.isnan(spikes).sum()/(len(spikes[0, :])))

session_df = pd.read_csv(os.path.join(session_dir, session + '_all_trials.csv'))
trial_t0 = np.array(session_df['time'])
durations = np.array(session_df['duration'])

fig, ax = plt.subplots(figsize=(200, 10))

for t_0, duration in zip(trial_t0, durations):
    plt.axvspan(t_0, t_0 + duration, alpha=0.2)

plt.plot(time, mean_act)
ax.set_xticks(np.arange(0, time[-1], 4))

plt.savefig('delete.png')