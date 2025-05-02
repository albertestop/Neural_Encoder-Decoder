import matplotlib.pyplot as plt
import torch
import pathlib
import numpy as np
import sys
import os
import random

project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.predictors import generate_predictors
from src import constants


def check_response(device, save_path):
    pth_files = [f for f in os.listdir(save_path) if f.endswith(".pth")]
    pth_files = sorted(pth_files)
    model_path = pth_files[-1]
    model_path = save_path + '/' + model_path

    data_dir = str(constants.sensorium_dir / constants.mice[0])
    trials = sorted(os.listdir(data_dir + '/data/responses'))
    random.seed(34)
    trial = random.choice(trials[int(len(trials) / 4) : - int(len(trials) / 4)])

    video = np.load(data_dir + f'/data/videos/{trial}')
    behavior = np.load(data_dir + f'/data/behavior/{trial}')
    pupil_center = np.load(data_dir + f'/data/pupil_center/{trial}')
    response = np.load(data_dir + f'/data/responses/{trial}')

    predictor = generate_predictors(
        device, [model_path], 1, [0]
    )
    with torch.no_grad():
        for n in range(len(predictor)):
            prediction = predictor[n].predict_trial(
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                mouse_index=0
            )
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs[0].imshow(prediction, aspect='auto', vmin=0, vmax=10)
    axs[0].set_title('Predicted responses')
    axs[1].imshow(response, aspect='auto', vmin=0, vmax=10)
    axs[1].set_title('Ground truth responses')
    fig.savefig(save_path + '/model_output.png')


if __name__ == '__main__':

    save_path = '/home/albertestop/Sensorium/data/experiments/train_test_000/fold_0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    check_response(device, save_path)
    