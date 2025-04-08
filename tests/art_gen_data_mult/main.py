import numpy as np
import os

old_path = '/home/albertestop/data/processed_data/sensorium_all_2023/2025-03-05_02_ESMT204_000_ART_GEN'
new_path = '/home/albertestop/data/processed_data/sensorium_all_2023/2025-03-05_02_ESMT204_001_ART_GEN'


for trial in os.listdir(os.path.join(new_path, 'data', 'videos')):
    responses = np.load(os.path.join(old_path, 'data', 'responses', trial))
    responses =  np.repeat(responses, repeats=500, axis=0)
    np.save(os.path.join(new_path, 'data', 'responses', trial), responses)
    print(trial)

#tiers = np.load(os.path.join(old_path, 'meta', 'trials', 'tiers.npy'))
#tiers = np.repeat(tiers, repeats=500, axis=0)
cell_coords = np.load(os.path.join(old_path, 'meta', 'neurons', 'cell_motor_coordinates.npy'))
cell_coords = np.zeros((len(responses), 3))
unit_ids = np.arange(0, len(responses))
unit_ids = np.repeat(unit_ids, repeats=500, axis=0)
#np.save(os.path.join(new_path, 'meta', 'trials', 'tiers.npy'), tiers)
np.save(os.path.join(new_path, 'meta', 'neurons', 'cell_motor_coordinates.npy'), cell_coords)
np.save(os.path.join(new_path, 'meta', 'neurons', 'unit_ids.npy'), unit_ids)
