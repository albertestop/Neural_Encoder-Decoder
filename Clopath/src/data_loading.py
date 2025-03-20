import random
import numpy as np
import argus

from Clopath.src.data_processing import Pipeline
from src.predictors import generate_predictors


def load_models(device, model_path, model_list, number_models, model):
    print('Loading reconstruction models')

    for n in range(len(model_list)):
        model[n] = argus.load_model(
            model_path[model_list[n]], device=device, optimizer=None, loss=None
            )
        model[n].eval()
        print(f'Model {model_path[model_list[n]]} loaded')

    predictor = generate_predictors(
        device, model_path, number_models, model_list
    )

    return model, predictor

"""print('Loading models for reconstruction: ', model_path[model_list[model_index]])
model = argus.load_model(
model_path[model_list[n]], device=device, optimizer=None, loss=None
)
model.eval()
predictor = Predictor(model_path=model_path[model_list[model_index]], device=device, blend_weights="ones")"""


def get_fold_data(mouse_data, json_data, mouse_key, fold_number, random_trials, start_trial, end_trial):
    # We will reconstruct the trials valid_trials[start_trial:end_trial] of fold fold_of_interest of mouse mice[mouse_index], N of trials = fold trials?---------
    trial_data_all = mouse_data['trials']
    fold_of_interest = str(fold_number)
    available_trials = json_data[mouse_key].get(fold_of_interest, [])
    if not available_trials:
        raise ValueError(f"No trials available for mouse {mouse_key} in fold {fold_of_interest} in JSON")
    trial_id_to_index = {trial_data['trial_id']: idx for idx, trial_data in enumerate(trial_data_all)}
    valid_trials = [t for t in available_trials if t in trial_id_to_index]
    if not valid_trials:
        raise ValueError(f"No valid trials exist both in JSON and trial_data_all for mouse {mouse_key} in fold {fold_of_interest}")
    if random_trials:
        trials_to_process = random.sample(valid_trials, min(end_trial - start_trial, len(valid_trials)))
    else:
        trials_to_process = valid_trials[start_trial:end_trial]
    return trial_data_all, available_trials, valid_trials, trials_to_process, trial_id_to_index


def load_trial_data(trial_data_all, model, device, trial_index, proc_params, length=None, print_dims=False):
    skip_frames = proc_params['load_skip_frames']
    trial_data = trial_data_all[trial_index]

    if length is None:
        length = trial_data["length"] 
    
    video = np.load(trial_data["video_path"])[..., skip_frames:skip_frames+length] # this has form hight, width, time
    behavior = np.load(trial_data["behavior_path"])[..., skip_frames:skip_frames+length]
    pupil_center = np.load(trial_data["pupil_center_path"])[..., skip_frames:skip_frames+length]
    responses = np.load(trial_data["response_path"])[..., skip_frames:skip_frames+length]

    pipeline = Pipeline(trial_index, device)
    inputs, video, behavior, pupil_center, responses, population_mask = pipeline(model[0], video, behavior, pupil_center, responses, proc_params)

    inputs = inputs.to(device)
    responses = responses.to(device)

    if print_dims:
        print("\nData info when reading it:")
        print(f"video.shape: {video.shape}" '\n'
            f"behavior.shape: {behavior.shape}" '\n'
            f"pupil_center.shape: {pupil_center.shape}"'\n'
            f"responses.shape: {responses.shape}")
        print(f"inputs.shape: {inputs.shape}")
        print(f"Population mask shape: {population_mask.shape}")
        
    return inputs, video, behavior, pupil_center, responses, population_mask, length