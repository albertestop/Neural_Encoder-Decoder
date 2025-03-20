import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter

import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)

"""
For 1 channel videos
"""

def apply_kernel(video, kernel):
    if kernel == 'luminiscence contrast':
        mean_fr_br = np.mean(video, axis=(0,1))
        return (video - mean_fr_br)/mean_fr_br

def rls_algorithm(video, kernel, responses, num_frames, dwns_l=17, beta=0.99, delta=0.00001, Q=450, tau=2):
    """
    Implements the Recursive Least Squares (RLS) algorithm to estimate the spatio-temporal kernel.

    Parameters:
    - feature_maps: 3D numpy array of shape (num_frames, grid_x, grid_y), containing the feature map sequences.
    - responses: 1D numpy array of length num_frames, containing the neural responses.
    - num_frames: Total number of stimulus frames.
    - subsample_grid: Subsample grid size (default is 17x17).
    - forgetting_factor: Forgetting factor for recursive update (default is 0.99).

    Returns:
    - weights: Estimated spatio-temporal kernel of shape (grid_x, grid_y).
    """
    # Initialize variables
    mu = 0
    w = np.zeros((dwns_l, dwns_l, num_frames))  # Kernel weights (flattened for computation)
    P = np.zeros((dwns_l, dwns_l, num_frames))
    P[:, :, 0] = video[:, :, 0] / delta  # Initial inverse correlation matrix
    video_fm = apply_kernel(video, kernel)
    u = subsample(video_fm)
    

    for n in range(1, num_frames):
        r_n = responses[n + tau]
        mu = beta * mu + (1 - beta)*r_n
        r_n = r_n - mu
        I = np.dot(u[n], P[:, :, n - 1])
        kappa = 1 + np.dot(I, u[n])
        k = np.dot(P[:, :, n - 1], u[n]) / (kappa)
        alpha = r_n - np.dot(w[n], u[n])
        w[n] = w[n - 1] + alpha*k
        T_1 = np.outer(k, I)
        P[:, :, n] = P[:, :, n - 1] - T_1
        if n % Q == 0:
            sigma = 0.4 + ((5 * Q) / n)
            w[n] = gaussian_filter(w[:, :, n], sigma=sigma)


# Example usage

sens_data_dir = '/home/albertestop/Sensorium/sensorium/data/sensorium_all_2023/dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20'
lab_data_dir = '/home/albertestop/Sensorium/sensorium/data/sensorium_all_2023/ESPM113_002'
trial = 0

num_frames = 1000
subsample_grid = 17
beta = 0.99
delta = 0.00001
kernel = 'luminiscence contrast'
feature_maps = np.random.randn(num_frames, subsample_grid, subsample_grid)  # Simulated feature maps
responses = np.random.randn(num_frames)  # Simulated neural responses

weights = rls_algorithm(feature_maps, responses, num_frames)
print("Estimated Kernel Weights:", weights)



correlations = np.array(correlations)
sens_corr = np.mean(correlations)
print('Finished with sensorium correlations, number of trials analyzed = ' + str(trial))


print('Mean correlation of sensorium responses: ' + str(sens_corr))
print('Mean correlation of lab responses: ' + str(lab_corr))

with open('plot/data/response_correlation/response_corr.txt' , "w") as file:
    file.write(f'Sens corr = ' + str(sens_corr) + '\n')
    file.write(f'Lab corr = ' + str(lab_corr) + '\n')
file.close()