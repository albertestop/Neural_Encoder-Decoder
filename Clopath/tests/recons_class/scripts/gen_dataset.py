from pathlib import Path
import os
import sys
import numpy as np

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent.parent
sys.path.append(str(parent_dir))

from Clopath.src.data_saving import generate_folder
from Clopath.tests.recons_class.src.metrics import *


exec_ids = [30, 31, 34, 36, 44, 47, 52, 54, 56, 60, 63, 66]
rec_good = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

mask = np.load('/home/albertestop/Sensorium/Clopath/reconstructions/masks/mask_m0.npy')

recons_dir = Path.cwd() / 'Clopath' / 'reconstructions' / 'results'

dataset = []

for exec, good in zip(exec_ids, rec_good):
    print('\nNew execution: ' + str(exec))
    recons_mouse = os.listdir(recons_dir / str(exec))[0]   # only works if reconstructions made for one mouse
    recons_ids = os.listdir(recons_dir / str(exec) / recons_mouse)
    for recons_id in recons_ids:
        print('Recons ' + str(recons_id))
        recons_df = []
        recons = np.load(recons_dir / str(exec) / Path(recons_mouse) / Path(recons_id) / 'reconstruction_array.npy')
        recons = np.transpose(recons[0, 0, :, 14:-14, :], axes=(1, 2, 0))
        n_nans = int(np.isnan(recons).sum() / (recons.shape[0] * recons.shape[1]))
        if n_nans > 0: recons = recons[:, :, :- n_nans]
        recons = np.transpose(recons, (2, 0, 1))
        
        mask = np.load('/home/albertestop/Sensorium/Clopath/reconstructions/masks/mask_m0.npy')
        mask = mask > 0.7
        mask = np.repeat(mask[None, 14:-14, :], recons.shape[0], axis=0)
        
        recons_df.append(str(exec) + '.' + str(recons_id))
        recons_df.append(temporal_corr(recons, mask))
        recons_df.append(temporal_ssim(recons, mask))
        recons_df.append(spectral_slope(recons, mask))
        recons_df.append(compression_gain(recons, mask))
        recons_df.append(good)

        dataset.append(recons_df)

    dataset_np = np.array(dataset)

dataset = np.array(dataset).astype(np.float32)
save_path = generate_folder(str(Path.cwd() / 'Clopath' / 'tests' / 'recons_class' / 'datasets'))
np.save(save_path + '/' + 'dataset.npy', dataset)
