import sys
from pathlib import Path
import os
import pandas as pd
import importlib
import numpy as np

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent.parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(parent_dir / 'data_processing'))
sys.path.append(str(parent_dir / 'Sensorium'))

import torch

from Clopath.tests.recons_class.src.model import ReconsClassModel
from Clopath.tests.recons_class.src.reconstruct import reconstruct, import_mask
from Clopath.src.data_saving import generate_folder
from Clopath.tests.recons_class.src.metrics import *
from Clopath.tests.recons_class.src.evaluator import Evaluator


from proc_scripts.config import *
from proc_resources import response_proc, behavior_proc, pupil_pos_proc
from proc_resources.proc_utils import *

# Session to reconstruct
animal = 'ESPM127'
session = '2025-04-01_01_ESPM127'
run_id = '000'
exp_directory = '/home/pmateosaparicio/data/Repository/'
proc_data_dir = Path('/home/albertestop/data/processed_data/sensorium_all_2023')
session_dir = os.path.join(exp_directory, animal, session)
rec_id = '30'   # One of the reconstruction runs
rec_config_path = Path(__file__).resolve().parent.parent.parent.parent / 'reconstructions' / 'results' / '30' / '2025-04-01_01_ESPM127_002' / '6' / 'config.py'
segment_length = 3
recons_path = '/home/albertestop/Sensorium/Clopath/tests/recons_class/data/reconstructions/' + session + '_' + run_id

# Encoding model
model_recons_class = 'train_test_000_18'
enc_path = "data/experiments/train_test_000_18/train_config.py"
enc_model_path = "/home/albertestop/Sensorium/data/experiments/train_test_000_18/fold_3/model-017-0.237781.pth"

file_path = Path(enc_path).expanduser().resolve()
spec = importlib.util.spec_from_file_location(
    name=file_path.stem,          # module name (anything unique; stem = filename without .py)
    location=file_path
)
second_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(second_module)

videos_df = pd.read_csv(os.path.join(session_dir, session + '_all_trials.csv'))
videos_df = videos_df[['time', 'duration', 'F1_name']]
videos_df['F1_name'] = trial_names(videos_df)

gen_reconstructions = True
if os.path.exists(recons_path):
    gen_reconstructions = False
else:
    gen_reconstructions = True
    os.mkdir(recons_path)

print('\nLoading data...')

responses = response_proc.Responses(response_params)
responses.load_data(data)

behavior = behavior_proc.Behavior(behavior_params, responses.time[-1])
behavior.load_data(data)

pupil_pos = pupil_pos_proc.PupilPosition(pupil_pos_params, responses.time[-1])
pupil_pos.load_data(data)

print('Data loaded.')

print('\nProcessing data...')

responses.process_global()
pupil_pos.process_global()
behavior.process_global()

print('Data processed.\n')

trials_t0 = videos_df['time'].iloc[0]
trials_t = np.arange(trials_t0, responses.time[-1], segment_length)
trials_i = np.arange(0, len(trials_t), 1)
trials_df = np.column_stack([trials_i, trials_t])

session_per = np.array(videos_df[['time', 'duration']])
evaluator = Evaluator(session + '_' + run_id, session_per, segment_length)


for i, t_i in trials_df:
    print('Processing segment ' + str(i) + '/' + str(len(trials_df)) + ' of the session.')
    segment_t = np.arange(t_i, t_i + 10, 1/30)

    if gen_reconstructions:
        responses.process(segment_t)

        behavior.process(segment_t)

        pupil_pos.process(segment_t)

        recons, mask = reconstruct(enc_model_path, rec_config_path, i, 0, responses.trial_data, behavior.trial_data, pupil_pos.trial_data, video_length = len(segment_t))
        np.save(recons_path + '/' + str(i) + '.npy', recons)

    else:
        recons = np.load(recons_path + '/' + str(i) + '.npy')
        mask = import_mask(rec_config_path)

    recons = np.transpose(recons[0, 0, :, 14:-14, :], axes=(1, 2, 0))
    n_nans = int(np.isnan(recons).sum() / (recons.shape[0] * recons.shape[1]))
    if n_nans > 0: recons = recons[:, :, :- n_nans]
    recons = np.transpose(recons, (2, 0, 1))
    
    mask = mask > 0.7
    mask = np.repeat(mask[None, 14:-14, :], recons.shape[0], axis=0)
    
    instance = [
        '0', 
        temporal_corr(recons, mask),
        temporal_ssim(recons, mask),
        spectral_slope(recons, mask),
        compression_gain(recons, mask)
    ]
    instance = np.array(instance).astype(np.float32)
    instance = instance.reshape(1, -1)

    model = ReconsClassModel(instance.shape[1])
    model.load_state_dict(torch.load('/home/albertestop/Sensorium/Clopath/tests/recons_class/models/clas_model.pth', weights_only=True))
    model.eval()
    with torch.no_grad():
        instance = torch.from_numpy(instance)
        prediction = model(instance)

    evaluator.iterate(t_i, prediction)

evaluator.save()