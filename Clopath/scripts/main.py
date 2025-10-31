import sys
import os
from pathlib import Path
import json
import numpy as np
import torch
import time
from tqdm import tqdm
import importlib.util

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(current_dir))

from src.data import get_mouse_data, save_fold_tiers
from src import constants
import Clopath.src.utils_reconstruction as utils
from Clopath.src.data_saving import *
from Clopath.src.data_loading import *
from Clopath.src.eval import Evaluator
from Clopath.src.predict import Predict
from Clopath.src.reconstruct import Reconstructor
from Clopath.reconstructions.masks.train_transparency_mask import train_mask

proc_config_path = "/home/albertestop/data/processed_data/sensorium_all_2023/" + constants.mice[0] + "/config.py"
spec = importlib.util.spec_from_file_location("proc_config", proc_config_path)
proc_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proc_config)


def reconstruct():
    import Clopath.scripts.config as dec_config
    strides_all, epoch_switch = utils.stride_calculator(dec_config.reconstruct_params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    execution_save_path = generate_folder(str(current_dir.parent) + dec_config.save_folder)

    print('execution_save_path: ' + execution_save_path)
    print('device: ' + str(device))
    print('\nStarting Reconstructions')

    model, predictor = load_models(
        device, dec_config.model_path, dec_config.model_list, dec_config.number_models, dec_config.model
    )

    video_correlations = []

    for mouse_index in dec_config.animals:

        mouse_save_path = execution_save_path + '/' + constants.mice[mouse_index]
        if not os.path.isdir(mouse_save_path): os.mkdir(mouse_save_path)
        mouse_key = constants.mice[mouse_index]
        with open(parent_dir / Path('Clopath') / dec_config.fold_file_path, 'r') as f:
            fold_data = json.load(f)
        mouse_data = get_mouse_data(mouse=mouse_key, splits=[dec_config.data_fold], s_type=proc_config.data['s_type'])
        """if proc_config.data['s_type'] in ('sleep', 'er', 'recons'):
            """
        mask_name = dec_config.pretrained_mask
        mask = np.load(parent_dir / Path(f'Clopath/reconstructions/masks/' + mask_name))
        """
        else:
            train_mask(dec_config.model_path[0], dec_config.animals, dec_config.data_fold)
            mask_name = 'mask_' + mouse_key + '.npy'
            mask = np.load(parent_dir / Path(f'Clopath/reconstructions/masks/' + mask_name))"""
        mask_update = torch.tensor(np.where(mask >= dec_config.mask_update_th, 1, 0)).to(device) # mask for gradients
        mask_eval = torch.tensor(np.where(mask >= dec_config.mask_eval_th, 1, 0)).to(device) # mask for pixels

        trial_data_all, available_trials, valid_trials, trials_to_process, trial_id_to_index = get_fold_data(
            mouse_data, fold_data, mouse_key, dec_config.fold_number, dec_config.random_trials, 
            dec_config.start_trial, dec_config.end_trial
        )

        print('Recostructing for :')
        print('Models ' + str(dec_config.model_list) + ' of the list: ' + str(dec_config.model_path))
        print('Dataset: ' +  str(constants.mice[mouse_index]))
        print('Fold: ' + str(dec_config.fold_of_interest))
        print('Trials: ' + str(valid_trials))
        
        print(f"\nMouse data:")
        print(f"Total mouse {mouse_key} trials in fold {dec_config.fold_of_interest}: {str(len(trial_data_all))}")
        print(f"N of trials = {len(available_trials)}. N of valid trials: {len(valid_trials)}")
        print(f"Trials to reconstruct = {valid_trials[dec_config.start_trial:dec_config.end_trial]}")
        print('Mask shape: ', mask.shape) # why has the mask 64x64 pixels?

        for trial in trials_to_process:

            # Trial data preparation

            torch.cuda.empty_cache()
            trial_index = trial_id_to_index[trial]
            trial_data = trial_data_all[trial_index]
            print(f'\nProcessing trial {trial}, index {trial_index}, mouse {mouse_key}')
            trial_save_path = mouse_save_path + '/' + str(trial)
            if not os.path.isdir(trial_save_path): os.mkdir(trial_save_path)
            inputs, video, behavior, pupil_center, responses, population_mask, video_length = load_trial_data(
                trial_data_all, model, device, trial_index, dec_config.proc_params, length=dec_config.video_length
            )

            # Reshape mask into 3-d to match trial video length
            mask_update_expanded = mask_update.repeat(1, 1, inputs.shape[1], 1, 1)
            mask_eval_expanded = mask_eval.repeat(1, 1, inputs.shape[1], 1, 1)

            # Neural responses to original video
            responses_predicted_original = Predict.predict(
                predictor, device, video, behavior, 
                pupil_center, mouse_index, video_length, stage='original')

            responses_predicted_mean = Predict.process_predictions(
                responses, responses_predicted_original, population_mask)

            # Neural responses of gray video
            responses_predicted_gray = Predict.predict(
                predictor, device, np.ones_like(video) * (255 / 2), behavior, 
                pupil_center, mouse_index, video_length, stage= 'gray')

            responses_predicted_gray = Predict.process_predictions(
                responses, responses_predicted_gray, population_mask,
                mask=True, mean=False)

            predictor_withgrads = [None] * dec_config.number_models
            for n in range(dec_config.number_models):
                predictor_withgrads[n] = utils.Predictor_JB(model[n], mouse_index, withgrads=True, dummy=False)


            # Reconstruct video

            print(f'\nRECONSTRUCTING MOUSE {mouse_key} TRIAL {trial}')
            if dec_config.opt_for_pred_responses: gt_responses = responses_predicted_mean.clone().detach()
            else: gt_responses = responses
            progress_bar = tqdm(range(epoch_switch[-1]))
            start_training_time = time.time()
            video_pred = utils.init_weights(inputs, dec_config.reconstruct_params['vid_init'], device)

            reconstructor = Reconstructor(
                device, dec_config.number_models, mask_update_expanded, population_mask,
                dec_config.reconstruct_params, epoch_switch, strides_all, gt_responses, 
                predictor_withgrads, inputs
            )

            evaluator = Evaluator(
                device, mouse_index, responses.shape, video_length, epoch_switch,
                population_mask, dec_config.number_models, mask_eval, trial_save_path, 
                predictor, dec_config.eval_params, video, behavior, pupil_center, responses, 
                responses_predicted_mean, responses_predicted_gray, current_dir
            )
            
            for i in progress_bar:

                video_pred, loss, gradients_fullvid = reconstructor.iterate(i, video_pred)
                
                evaluator.iterate(i, loss, progress_bar, video_pred, gradients_fullvid)

            end_training_time = time.time()
            training_time = end_training_time - start_training_time
            print(f"Training time for trial {trial}: {training_time:.2f} seconds")

            
            # Save Results

            evaluator.save_results(
                strides_all, trial, mask, training_time, video_pred.cpu().detach().numpy()
            )
            mp4_path = reconstructor.reconstruct_video(trial_save_path, dec_config.smooth, evaluator.concat_video)
            video_correlations.append(evaluator.video_corr[-1])

            print(f"\nReconstruction completed for mouse {mouse_key}, trial {trial}")
            print(f"Saved reconstruction summary for mouse {mouse_key}, trial {trial}")
            print(f"Model used: {dec_config.model_path[dec_config.model_list[0]]}")
            print(f"Results saved in: {trial_save_path}")
            print(f"Reconstruction video saved as: {mp4_path}")

        print(video_correlations)

        if proc_config.data['s_type'] not in ('sleep', 'er'):
            with open(os.path.join(mouse_save_path, "rec_perf.txt"), "w") as file:
                file.write('Reconstruction mean corr: ' + str(np.array(video_correlations).mean()))

    if proc_config.data['s_type'] in ('sleep', 'er'):
        reconstructor.reconstruct_whole_session(mouse_save_path, proc_config)

    print("Reconstruction process completed for all mice and trials.")
    return np.array(video_correlations), execution_save_path


if __name__ == "__main__":
    video_correlations, save_path = reconstruct()