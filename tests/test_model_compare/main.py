import json
import torch
import pathlib
import numpy as np
import sys
import os

project_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.predictors import generate_predictors
from src.metrics import corr


def test_model(device, path, mouse_key):
    fold_of_interest = path[-1]
    pth_files = [f for f in os.listdir(path) if f.endswith(".pth")]
    pth_files = sorted(pth_files)
    model_path = pth_files[-1]
    model_path = str(path) + '/' + model_path

    data_dir = '/home/albertestop/data/processed_data/sensorium_all_2023' + mouse_key
    with open(project_root / pathlib.Path('Clopath') / 'folds_trials.json', 'r') as f:
        json_data = json.load(f)
    trials = json_data[mouse_key[1:]].get(fold_of_interest, [])

    predictor = generate_predictors(
        device, [model_path], 1, [0]
    )

    resp_corr = []

    print(f'\nTesting model {model_path}, with fold {fold_of_interest}, of mouse {mouse_key}')

    for trial in trials[:1]:
        print(trial)

        video = np.load(data_dir + f'/data/videos/{trial}.npy')
        behavior = np.load(data_dir + f'/data/behavior/{trial}.npy')
        pupil_center = np.load(data_dir + f'/data/pupil_center/{trial}.npy')
        response = np.load(data_dir + f'/data/responses/{trial}.npy')

        if np.isnan(video).any():
            nan_idx = sorted(np.flatnonzero(np.isnan(response)))[0]
            video = video[:, :, :nan_idx]
            response = response[:, :nan_idx]
            pupil_center = pupil_center[:, :nan_idx]
            behavior = behavior[:, :nan_idx]
    
        with torch.no_grad():
            for n in range(len(predictor)):
                prediction = predictor[n].predict_trial(
                    video=video,
                    behavior=behavior,
                    pupil_center=pupil_center,
                    mouse_index=0
                )
        
        

    perf = np.mean(resp_corr)
    print(perf/1.4286)


if __name__ == '__main__':

    model_path = '/home/albertestop/Sensorium/data/experiments/train_test_000_13/fold_0'
    data_dir = '/dynamic29515_ART_FROM_VIDEO_7'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_model(device, model_path, data_dir)