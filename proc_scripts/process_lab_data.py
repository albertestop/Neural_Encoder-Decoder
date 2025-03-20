import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from src import constants
from configs.data_proc_001 import *
from proc_resources import response_proc, video_proc, behavior_proc, pupil_pos_proc
from proc_resources.sanity_check import sanity_check
from proc_resources.proc_utils import *


trials_df = pd.read_csv(os.path.join(data['session_dir'], data['session'] + '_all_trials.csv'))
trials_df = trials_df[['time', 'duration', 'F1_name']]
trials_df['F1_name'] = trial_names(trials_df)

videos = video_proc.Video(videos_params)

responses = response_proc.Responses(response_params)
responses.load_data(data)

behavior = behavior_proc.Behavior(behavior_params, responses.time[-1])
behavior.load_data(data)

pupil_pos = pupil_pos_proc.PupilPosition(pupil_pos_params, responses.time[-1])
pupil_pos.load_data(data)

j = 0
mouse = data['session'] + '_' + data['mouse_run']
mouse_dir = constants.sensorium_dir / mouse

print('\nSaving data in ' + str(os.path.join(mouse_dir, 'data/videos')))
if os.path.isdir(os.path.join(mouse_dir, 'data/videos')):
    print("The directory where the data is going to be saved already exists.")
    print("The execution will overwrite that directory and its contents.")
    carry_on = input("Continue with the execution? Y/N\n")
    if carry_on != 'Y':
        exit()

os.makedirs(os.path.join(mouse_dir, 'data/videos'), exist_ok=True)
os.makedirs(os.path.join(mouse_dir, 'data/responses'), exist_ok=True)
os.makedirs(os.path.join(mouse_dir, 'data/behavior'), exist_ok=True)
os.makedirs(os.path.join(mouse_dir, 'data/pupil_center'), exist_ok=True)

print('\nData loaded.')

sanity_check(
    parent_dir, trials_df, responses, behavior, pupil_pos, videos
)

print('Sanity check completed, see .../src/data/data_processing/sanity_checks/session_id for further info\n')

for trial_id, trial_t0, i in zip(trials_df['F1_name'], trials_df['time'], np.arange(len(trials_df['time']))):
    print(str(i) + '/' + str(len(trials_df['F1_name'])))
    print('Processing trial ' + trial_id + ' of the experiment.')
    
    
    videos.load_video(trial_id, trial_t0)
    while(videos.trial_video.shape[2] != len(videos.trial_frame_time)):
        videos.trial_frame_time = videos.trial_frame_time[:-1]
    
    responses.process(videos.trial_frame_time)

    behavior.process(videos.trial_frame_time, videos.trial_video)

    pupil_pos.process(videos.trial_frame_time)


    num_windows, window_indices = split_trial(len(behavior.trial_data[0, :]))
    
    for i in range(num_windows):
        filepath_videos = os.path.join(mouse_dir, 'data/videos', f"{j}.npy")
        filepath_response = os.path.join(mouse_dir, 'data/responses', f"{j}.npy")
        filepath_behavior = os.path.join(mouse_dir, 'data/behavior', f"{j}.npy")
        filepath_pupil_pos = os.path.join(mouse_dir, 'data/pupil_center', f"{j}.npy")
        np.save(filepath_videos, videos.trial_video[:, :, window_indices[i, 0]:window_indices[i, 1]])
        np.save(filepath_response, responses.trial_data[:, window_indices[i, 0]:window_indices[i, 1]])
        np.save(filepath_behavior, behavior.trial_data[:, window_indices[i, 0]:window_indices[i, 1]])
        np.save(filepath_pupil_pos, pupil_pos.data[:, window_indices[i, 0]:window_indices[i, 1]])

        j += 1

    print('Finished processing data.')    
    print('Trial data saved to: ' + str(mouse_dir) + ' \n')


# Create tiers.py
num_live_test = int(j * 0)
num_train = j - num_live_test
tiers = np.array(['train'] * num_train + ['live_test_main'] * num_live_test)
np.random.shuffle(tiers)
os.makedirs(os.path.join(mouse_dir, 'meta/trials/'), exist_ok=True)
np.save(os.path.join(mouse_dir, 'meta/trials/tiers.npy'), tiers)
print('Tiers file created.')

# Create cell_motor_coordinates.py
cell_motor_coords = np.zeros((responses.num_neurons, 3))
os.makedirs(os.path.join(mouse_dir, 'meta/neurons/'), exist_ok=True)
np.save(os.path.join(mouse_dir, 'meta/neurons/cell_motor_coordinates.npy'), cell_motor_coords)
print('Neuron coords file created.')

# Create unit_ids.py
unit_ids = np.arange(responses.num_neurons)
np.save(os.path.join(mouse_dir, 'meta/neurons/unit_ids.npy'), unit_ids)
print('Unit ids file created.')