from pathlib import Path
import numpy as np
import importlib

from Clopath.src.data_processing import Pipeline

def load_trial_data(model, trial_index, responses, behavior, pupil_center, device, length, proc_params):

    video = np.ones((64, 64, length))
    video = video*(255/2)

    pipeline = Pipeline(trial_index, device)
    inputs, video, behavior, pupil_center, responses, population_mask = pipeline(model[0], video, behavior, pupil_center, responses, proc_params)

    inputs = inputs.to(device)
    responses = responses.to(device)
        
    return inputs, behavior, pupil_center, responses, population_mask


def load_config(path):
    spec = importlib.util.spec_from_file_location("rec_config", path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def cut_exp_mask(mask, thr, temp_l):
    cols_with_large_values = np.any(mask > thr, axis=0)
    mask_x_min = np.where(cols_with_large_values)[0][0]
    mask_x_max = np.where(cols_with_large_values)[0][-1]
    rows_with_large_values = np.any(mask > thr, axis=1)
    mask_y_min = np.where(rows_with_large_values)[0][0]
    mask_y_max = np.where(rows_with_large_values)[0][-1]
    mask = mask[mask_y_min:mask_y_max + 1, mask_x_min:mask_x_max + 1]
    mask = np.repeat(mask[None, :, :], temp_l, axis=0)
    mask = np.repeat(mask, 10, axis=1)
    mask = np.repeat(mask, 10, axis=2)
    return mask


def load_mask(rec_config, proc_config, session):
    """
    If not movie session mask used was determined in config
    If movie session: mask was trained before reconstructing and saved with session name
    """
    if proc_config.data['s_type'] in ('sleep', 'er', 'recons'):
        mask_name = rec_config.pretrained_mask
    else:
        mask_name = 'mask_' + session + '.npy'
    
    mask = np.load('/home/albertestop/Sensorium/Clopath/reconstructions/masks/' + mask_name)[14:-14, :]
    return mask
    

def read_session(dir_path: str) -> str:
    p = Path(dir_path)
    folders = [d for d in p.iterdir() if d.is_dir()]
    if len(folders) != 1:
        raise ValueError(f"Expected exactly 1 folder in {p}, found {len(folders)}")
    return folders[0].name