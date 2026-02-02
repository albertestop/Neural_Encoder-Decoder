import sys
from pathlib import Path
import numpy as np
import torch
import time
from tqdm import tqdm
import importlib

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent.parent
sys.path.append(str(parent_dir))

import Clopath.src.utils_reconstruction as utils
from Clopath.src.data_saving import *
from Clopath.src.data_loading import load_models
from Clopath.src.reconstruct import Reconstructor
from Clopath.tests.recons_class.src.data_loading import load_trial_data


def import_mask(rec_config_path):
    file_path = Path(rec_config_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    rec_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec_config)
    
    mask = np.load(parent_dir / Path(f'Clopath/reconstructions/masks/' + rec_config.mask_name))
    return mask


def reconstruct(enc_model_path, rec_config_path, trial_index, mouse_index, responses, behavior, pupil_center, video_length):

    # Import recons config

    file_path = Path(rec_config_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    rec_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rec_config)
    
    strides_all, epoch_switch = utils.stride_calculator(rec_config.reconstruct_params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, predictor = load_models(
        device, [enc_model_path], [0], 1, [None]
    )

    mask = np.load(parent_dir / Path(f'Clopath/reconstructions/masks/' + rec_config.mask_name))
    mask_update = torch.tensor(np.where(mask >= rec_config.mask_update_th, 1, 0)).to(device) # mask for gradients

    # Trial data preparation

    torch.cuda.empty_cache()
    inputs, behavior, pupil_center, responses, population_mask = load_trial_data(
        model, trial_index, responses, behavior, pupil_center, device, video_length, rec_config.proc_params
    )

    # Reshape mask into 3-d to match trial video length
    mask_update_expanded = mask_update.repeat(1, 1, inputs.shape[1], 1, 1)

    predictor_withgrads = [None] * 1
    for n in range(1):
        predictor_withgrads[n] = utils.Predictor_JB(model[n], mouse_index, withgrads=True, dummy=False)


    # Reconstruct video

    print(f'\nRECONSTRUCTING SESSION WINDOW')
    gt_responses = responses
    progress_bar = tqdm(range(epoch_switch[-1]))
    video_pred = utils.init_weights(inputs, rec_config.reconstruct_params['vid_init'], device)

    reconstructor = Reconstructor(
        device, rec_config.number_models, mask_update_expanded, population_mask,
        rec_config.reconstruct_params, epoch_switch, strides_all, gt_responses, 
        predictor_withgrads, inputs
    )

    for i in progress_bar:

        video_pred, _, _ = reconstructor.iterate(i, video_pred)

    return video_pred.detach().cpu().numpy(), mask  



if __name__ == '__main__':
    enc_model_path = '/home/albertestop/Sensorium/data/experiments/train_test_000_18/fold_3/model-017-0.237781.pth'
    rec_config_path = '/home/albertestop/Sensorium/Clopath/reconstructions/results/30/2025-04-01_01_ESPM127_002/6/config.py'
    mouse_index = 0
    responses = np.load('/home/albertestop/data/processed_data/sensorium_all_2023/2025-04-01_01_ESPM127_002/data/responses/0.npy')
    behavior = np.load('/home/albertestop/data/processed_data/sensorium_all_2023/2025-04-01_01_ESPM127_002/data/behavior/0.npy')
    pupil_center = np.load('/home/albertestop/data/processed_data/sensorium_all_2023/2025-04-01_01_ESPM127_002/data/pupil_center/0.npy')
    video_length = 300
    reconstruct(enc_model_path, rec_config_path, 0, mouse_index, responses, behavior, pupil_center, video_length)
    
