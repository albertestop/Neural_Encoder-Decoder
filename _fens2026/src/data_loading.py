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