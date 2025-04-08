import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)

def load_sens_comp_data():

    sens_data_dir = '/home/albertestop/data/processed_data/sensorium_all_2023/dynamic29515'

    trial_sens = '700.npy'

    sens_video = np.load(os.path.join(sens_data_dir, 'data', 'videos', trial_sens))
    sens_response = np.load(os.path.join(sens_data_dir, 'data', 'responses', trial_sens))
    sens_behavior = np.load(os.path.join(sens_data_dir, 'data', 'behavior', trial_sens))
    sens_pupil_center = np.load(os.path.join(sens_data_dir, 'data', 'pupil_center', trial_sens))

    return sens_video, sens_response, sens_behavior, sens_pupil_center