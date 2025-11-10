import os
import numpy as np
import pickle

def plane_correlations(save_dir, data_path):

    with open(os.path.join(data_path, 'recordings', 's2p_ch0.pickle'), 'rb') as file:
        file = pickle.load(file)
    neuron_planes = file['Depths']

    data = np.load(os.path.join(save_dir, 'responsiveness/neuron_corr.npy'))

    with open(os.path.join(save_dir, 'plane_correlations.txt'), "w") as file:
        for i in range(int(neuron_planes[-1, 0] + 1)):
            indexes = neuron_planes[neuron_planes == i]
            plane_perf = data[indexes[:]]
            file.write(f'\nPlane {i} performance: {plane_perf.mean()}')