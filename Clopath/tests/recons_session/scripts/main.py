import os
import sys
from pathlib import Path
import numpy as np

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent.parent
sys.path.append(str(parent_dir))

from Clopath.tests.recons_session.src.reconstruct import *
from Clopath.tests.recons_session.src.build import *
from Clopath.tests.recons_session.src.data_loading import *

def recons_session(session, r_type, mask, recons_run):

    recons_path = Path('/home/albertestop/Sensorium/Clopath/reconstructions/results/' + recons_run + '/' + session)
    mask = np.load('/home/albertestop/Sensorium/Clopath/reconstructions/masks/' + mask + '.npy')[14:-14, :]

    segments = [name for name in os.listdir(recons_path)
        if os.path.isdir(os.path.join(recons_path, name))]
    segments = sorted(segments, key=int)
    segments = segments[:-1]

    recons_config_path = str(recons_path) + '/' + segments[0] + '/config.py'
    rec_config = load_config(recons_config_path)

    t_0 = 0
    t_f = t_0 + (len(segments) * 10)
    fr_i = t_0 * 30
    fr_f = fr_i + (len(segments) * 10 * 30)
    recons_time = np.arange(fr_i, fr_f) / 30

    recons = build_recons(recons_path, mask, recons_time)

    if r_type == 'movie':
        proc_config_path = "/home/albertestop/data/processed_data/sensorium_all_2023/" + session + "/config.py"
        proc_config = load_config(proc_config_path)

        build_movie(proc_config, rec_config, mask, recons_time, recons_path, t_0, t_f)


if __name__ == '__main__':
    session = '2025-07-04_04_ESPM154_004_concat'
    r_type = 'movie'
    mask = 'mask_2025-07-04_04_ESPM154_004'
    recons_run = '173'
    recons_session(session, r_type, mask, recons_run)