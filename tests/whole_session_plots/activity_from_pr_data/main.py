import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from pathlib import Path

dataset = "2025-08-07_05_ESPM163_er2"
pr_session_dir = "/home/albertestop/data/processed_data/sensorium_all_2023/" + dataset 
data_path = "/home/albertestop/data/processed_data/sensorium_all_2023/" + dataset + "/data/responses/"
animal = 'ESPM163'
session = '2025-08-07_05_ESPM163'
exp_directory = '/home/melinatimplalexi/data/Repository/'
session_dir = str(os.path.join(exp_directory, animal, session))

segments = os.listdir(data_path)
segments = sorted(segments, key=lambda s: int(Path(s).stem))

session_activity = np.array([])

for segment in segments:
    trial_resp = np.load(data_path + '/' + segment)
    trial_resp = trial_resp[:, 30:-30]
    trial_resp = trial_resp.mean(axis=0)
    session_activity = np.concatenate((session_activity, trial_resp))

session_df = pd.read_csv(os.path.join(session_dir, session + '_all_trials.csv'))
trial_t0 = np.array(session_df['time'])
durations = np.array(session_df['duration'])
time = np.arange(0, session_df['time'].iloc[-1], 1/30)

fig, ax = plt.subplots(figsize=(200, 10))

for t_0, duration in zip(trial_t0, durations):
    plt.axvspan(t_0, t_0 + duration, alpha=0.2)

plt.plot(time[:len(session_activity)], session_activity)
ax.set_xticks(np.arange(0, time[-1], 4))

plt.savefig('delete2.png')