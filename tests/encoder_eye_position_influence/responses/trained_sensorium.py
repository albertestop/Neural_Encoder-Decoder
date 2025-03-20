import sys
from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent
sys.path.append(str(parent_dir))

import argus
from src.data import get_mouse_data
from src import constants
from src.predictors import generate_predictors
from Clopath.src.data_saving import *
from Clopath.src.data_processing import *
from Clopath.src.data_loading import *
import Clopath.src.utils_reconstruction as utils
from config import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f'Model list: {model_list}')
print(f'Number of models: {number_models}')
print('device: ' + str(device))

## Loading of pretrained encoding models (sensorium: video->neural activity)
for n in range(len(model_list)):
    model[n] = argus.load_model(
        model_path[model_list[n]], device=device, optimizer=None, loss=None
        )
    model[n].eval()

predictor = generate_predictors(
    opt_for_pred_responses, device,
    model_path, number_models, model_list)

json_file_path = Path.joinpath(parent_dir, Path('Clopath/folds_trials.json'))
with open(json_file_path, 'r') as f:
    json_data = json.load(f)

print('\nStarting response predictions')

execution_save_path = generate_folder(str(current_dir.parent) + save_folder)

for i, model_index in enumerate(model_list):
    model_save_path = execution_save_path + '/' + model_name[model_index]
    if not os.path.isdir(model_save_path): os.mkdir(model_save_path)
    mean_response_corr = []
    mean_response_corr_no_eye = []
    mean_response_corr_no_pxl = []
    mean_no_pxl_no_eye_corr = []

    for mouse_index in animals:
        mouse_save_path = model_save_path + '/' + constants.mice[mouse_index]
        if not os.path.isdir(mouse_save_path): os.mkdir(mouse_save_path)

        mouse_key = constants.mice[mouse_index]
        mouse_data = get_mouse_data(mouse=mouse_key, splits=[data_fold])
        trial_data_all, available_trials, valid_trials, trials_to_process, trial_id_to_index = get_fold_data(mouse_data, json_data, mouse_key, fold_number, random_trials, start_trial, end_trial)

        print(f"\nMouse data:")
        print(f"Total mouse {mouse_key} trials: {str(len(trial_data_all))}")
        print(f"Using trials in fold_{fold_of_interest} for mouse {mouse_key} defined in folds_trials_new_mice.json. N of trials = {len(available_trials)}")
        print(f"N of valid trials: {len(valid_trials)}")

        # For each trial we will do:
        for trial in trials_to_process:
            torch.cuda.empty_cache()
            trial_index = trial_id_to_index[trial]
            trial_data = trial_data_all[trial_index]
            print(f'\nProcessing trial {trial}, index {trial_index}, mouse {mouse_key}, model {model_name[model_index]}')

            trial_save_path = mouse_save_path + '/' + str(trial)
            if not os.path.isdir(trial_save_path): os.mkdir(trial_save_path)

            if video_length is None:
                video_length = trial_data["length"]
            
            
            proc_params['video_params']['pixel_supression'] = 0
            _, video, behavior, pupil_center, responses, population_mask, video_length = load_trial_data(
                trial_data_all, model, device, trial_index, proc_params, length=video_length)

            responses_predicted = torch.zeros((constants.num_neurons[mouse_index], video_length), device=device)
            prediction = predictor[i].predict_trial(
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                mouse_index=mouse_index
            )
            responses_predicted[:, :] = torch.from_numpy(prediction).to(device)


            proc_params['video_params']['pixel_supression'] = 0
            proc_params['pc_params']['pupil_pos'] = 'mean'
            _, video, behavior, pupil_center, responses, population_mask, video_length = load_trial_data(
                trial_data_all, model, device, trial_index, proc_params, length=video_length)

            responses_predicted_no_eye = torch.zeros((constants.num_neurons[mouse_index], video_length), device=device)
            prediction = predictor[i].predict_trial(
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                mouse_index=mouse_index
            )
            responses_predicted_no_eye[:, :] = torch.from_numpy(prediction).to(device)


            proc_params['pc_params']['pupil_pos'] = 'original'
            proc_params['video_params']['pixel_supression'] = 1
            _, video, behavior, pupil_center, responses, population_mask, video_length = load_trial_data(
                trial_data_all, model, device, trial_index, proc_params, length=video_length)

            responses_predicted_pxl_supr = torch.zeros((constants.num_neurons[mouse_index], video_length), device=device)#----initialize the response tensor
            prediction = predictor[i].predict_trial(
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                mouse_index=mouse_index
            )
            responses_predicted_pxl_supr[:, :] = torch.from_numpy(prediction).to(device)
            proc_params['video_params']['pixel_supression'] = 1


        
            #-------------------Results----------------------

            save_path = execution_save_path + '/' + str(mouse_index) + '_' + str(trial)  + '_' + str(model_name[model_index])

            response_corr = utils.compute_corr(
                responses.cpu().detach().numpy(),
                responses_predicted.cpu().detach().numpy()
            )
            response_corr_no_eye = utils.compute_corr(
                responses.cpu().detach().numpy(),
                responses_predicted_no_eye.cpu().detach().numpy()
            )
            response_corr_no_pxl = utils.compute_corr(
                responses.cpu().detach().numpy(),
                responses_predicted_pxl_supr.cpu().detach().numpy()
            )
            no_pxl_no_eye_corr = utils.compute_corr(
                responses_predicted_pxl_supr.cpu().detach().numpy(),
                responses_predicted_no_eye.cpu().detach().numpy()
            )

            fig, axs = plt.subplots(3, 2, figsize=(20, 20))
            fig.suptitle(f'mouse {mouse_index} trial {trial} model {model_list[0]}, {model_name[model_list[0]]}', fontsize=16)
            axs[0, 0].plot(pupil_center[:, :].T)
            axs[0, 0].set_title('Pupil position')
            axs[0, 1].imshow(responses.cpu().detach().numpy(), aspect='auto', vmin=0, vmax=10)
            axs[0, 1].set_title('GT Responses')
            axs[1, 0].imshow(responses_predicted.cpu().detach().numpy(), aspect='auto', vmin=0, vmax=10)
            axs[1, 0].set_title('Predicted Responses')
            axs[1, 1].imshow(responses_predicted_no_eye.cpu().detach().numpy(), aspect='auto', vmin=0, vmax=10)
            axs[1, 1].set_title('Predicted Responses No Pupil Position')
            axs[2, 0].imshow(responses_predicted_pxl_supr.cpu().detach().numpy(), aspect='auto', vmin=0, vmax=10)
            axs[2, 0].set_title('Predicted Responses ' + str(proc_params['video_params']['pixel_supression']) + ' Pixel Missing')
            fig.savefig(trial_save_path + '/results.png')

            with open(trial_save_path + '/results.txt', "w") as file:
                file.write("\nGT Response correlation with response predictions: " + str(response_corr))
                file.write("\nGT Response correlation with response predictions made with no eye data: " + str(response_corr_no_eye))
                file.write("\nGT Response correlation with response predictions made with " + str(proc_params['video_params']['pixel_supression']) + " missing pixel: " + str(response_corr_no_pxl))
                file.write("\nResponse predictions made with no eye data correlation with response predictions made with " + str(proc_params['video_params']['pixel_supression']) + " missing pixel: " + str(no_pxl_no_eye_corr))
            file.close()

            mean_response_corr.append(response_corr)
            mean_response_corr_no_eye.append(response_corr_no_eye) 
            mean_response_corr_no_pxl .append(response_corr_no_pxl)
            mean_no_pxl_no_eye_corr.append(no_pxl_no_eye_corr)


    with open(model_save_path + '/results.txt', "w") as file:
        file.write("\nGT Response correlation with response predictions: " + str(np.array(mean_response_corr).mean()))
        file.write("\nGT Response correlation with response predictions made with no eye data: " + str(np.array(mean_response_corr_no_eye).mean()))
        file.write("\nGT Response correlation with response predictions made with " + str(proc_params['video_params']['pixel_supression']) + " missing pixel: " + str(np.array(mean_response_corr_no_pxl).mean()))
        file.write("\nResponse predictions made with no eye data correlation with response predictions made with " + str(proc_params['video_params']['pixel_supression']) + " missing pixel: " + str(np.array(mean_no_pxl_no_eye_corr).mean()))
    file.close()
    
