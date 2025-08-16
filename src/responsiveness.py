
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

sys.path.append(str(Path.home() / Path('Sensorium')))

from src.predictors import generate_predictors


def responsiveness(model_dir, dataset_id):

    save_path = os.path.join(model_dir, 'responsiveness')

    os.makedirs(save_path, exist_ok=True)

    data_dir = str(Path.home()) + '/data/processed_data/sensorium_all_2023/' + dataset_id
    with open(str(Path.home()) + '/Sensorium/Clopath/folds_trials.json', 'r') as f:
        fold_data = json.load(f)
    trials = fold_data[dataset_id].get('0', [])

    model_name = [f for f in os.listdir(model_dir) if f.endswith(".pth")][0]
    model_path = str(model_dir) + '/' + model_name

    predictor = generate_predictors(
            "cuda:0", [model_path], 1, [0]
        )

    correlations = []

    for trial in trials:

        video = np.load(data_dir + f'/data/videos/{trial}.npy')
        behavior = np.load(data_dir + f'/data/behavior/{trial}.npy')
        pupil_center = np.load(data_dir + f'/data/pupil_center/{trial}.npy')
        response = np.load(data_dir + f'/data/responses/{trial}.npy')

        with torch.no_grad():
            for n in range(len(predictor)):
                prediction = predictor[n].predict_trial(
                    video=video,
                    behavior=behavior,
                    pupil_center=pupil_center,
                    mouse_index=0
                )

        correlations_trial = []

        for neuron in range(prediction.shape[0]):

            corr = np.corrcoef(response[neuron], prediction[neuron])[0, 1]
            
            correlations_trial.append(corr)

        correlations.append(correlations_trial)

    correlations = np.array(correlations)
    correlations = np.mean(correlations, axis=0)
    
    mean_corr = correlations.mean()
    corr_10_idx = np.where(correlations > 0.1)
    corr_10 = len(correlations[corr_10_idx])
    np.save(os.path.join(save_path, 'corr_gr_10.npy'), corr_10_idx)
    corr_25_idx = np.where(correlations > 0.25)
    corr_25 = len(correlations[corr_25_idx])
    np.save(os.path.join(save_path, 'corr_gr_25.npy'), corr_25_idx)
    corr_40_idx = np.where(correlations > 0.40)
    corr_40 = len(correlations[corr_40_idx])
    np.save(os.path.join(save_path, 'corr_gr_40.npy'), corr_40_idx)
    corr_55_idx = np.where(correlations > 0.55)
    corr_55 = len(correlations[corr_55_idx])
    np.save(os.path.join(save_path, 'corr_gr_55.npy'), corr_55_idx)
    corr_70_idx = np.where(correlations > 0.70)
    corr_70 = len(correlations[corr_70_idx])
    np.save(os.path.join(save_path, 'corr_gr_70.npy'), corr_70_idx)

    plt.cla()
    plt.clf()
    plt.hist(correlations, bins=50, edgecolor='black')
    plt.xlabel('Prediciton corr')
    plt.ylabel('N neurons')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'response_study.png'))
    plt.close()
    plt.cla()

    with open(os.path.join(save_path, 'neural_response_study.txt'), "w") as file:
        file.write('Encoder correlation: ' + str(model_name[:-4]))
        file.write('\nMean neural encoder correlation: ' + str(mean_corr))
        file.write('\nN of neurons with corr > 10: ' + str(corr_10))
        file.write('\nN of neurons with corr > 25: ' + str(corr_25))
        file.write('\nN of neurons with corr > 40: ' + str(corr_40))
        file.write('\nN of neurons with corr > 55: ' + str(corr_55))
        file.write('\nN of neurons with corr > 70: ' + str(corr_70))

if __name__ == '__main__':
    
    model_dir = '/home/albertestop/Sensorium/data/experiments/new_experiment/fold_0/'
    dataset_id = '2025-04-01_01_ESPM127_005'

    responsiveness(model_dir, dataset_id)