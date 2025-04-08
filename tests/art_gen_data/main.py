import numpy as np
import os

experiment_path = '/home/albertestop/data/processed_data/sensorium_all_2023/2025-03-05_02_ESMT204_000_ART_GEN'


for trial in os.listdir(os.path.join(experiment_path, 'data', 'videos')):
    video = np.load(os.path.join(experiment_path, 'data', 'videos', trial))
    responses = np.load(os.path.join(experiment_path, 'data', 'responses', trial))
    video_brighness = np.mean(np.mean(video, axis=0), axis=0)
    resp_max = np.max(responses)
    video_brighness = (video_brighness/np.max(video_brighness))*(resp_max)
    responses[:] = video_brighness
    np.save(os.path.join(experiment_path, 'data', 'responses', trial), responses)
    print(trial)
