import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from configs.data_proc_001 import data
from proc_resources.proc_utils import trial_indexes
from sc_resources.sc_utils import *
from sc_resources.responses_utils import *
from sc_resources.speed_utils import *
from sc_resources.pupil_utils import *
from sc_resources.sens_utils import *

def sanity_check(parent_dir, trials_df, responses, behavior, pupil_pos, videos):

    print('\nRunning Sanity Check')

    os.makedirs(os.path.join(parent_dir, 'pre_processing', data['session'], 'sanity_check'), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, 'pre_processing', data['session'], 'sanity_check', 'responses'), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, 'pre_processing', data['session'], 'sanity_check', 'speed'), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, 'pre_processing', data['session'], 'sanity_check', 'pupil'), exist_ok=True)

    sens_video, sens_response, sens_behavior, sens_pupil_dilation = load_sens_comp_data()

    sc_time_coherence(trials_df['time'], responses, behavior, pupil_pos)
    full_session_sc(parent_dir, data, responses, behavior, pupil_pos)


    # RESPONSES
    responses_utils = ResponsesSC(parent_dir, data['session'])

    responses_utils.sc_plane_neurons(responses)
    responses_utils.responses_time_coherence(trials_df['time'], responses)
    tr_indexes = trial_indexes(np.array(trials_df['time'])[0], np.array(trials_df['duration'])[0], responses.time_global)
    responses_utils.sens_lab_comparison(sens_response, responses.data[:, tr_indexes])


    # SPEED
    speed_utils = SpeedSC(parent_dir, data['session'])

    speed_utils.speed_time_coherence(trials_df['time'], behavior)

    # PUPIL
    pupil_utils = PupilSC(parent_dir, data['session'])

    pupil_utils.pupil_dilation_time_coherence(trials_df['time'], behavior)
    tr_indexes = trial_indexes(np.array(trials_df['time'])[0], np.array(trials_df['duration'])[0], behavior.pupil_dilation_time)
    videos.load_video(trials_df['F1_name'].iloc[0], trials_df['time'].iloc[0], trials_df['duration'].iloc[0])
    pupil_utils.pupil_dilation_sens_comp(sens_behavior, behavior.pupil_dilation[tr_indexes], videos.trial_video)

    pupil_utils.pupil_pos_time_coherence(trials_df['time'], pupil_pos)
    tr_indexes = trial_indexes(np.array(trials_df['time'])[0], np.array(trials_df['duration'])[0], pupil_pos.time)
    pupil_utils.pupil_pos_sens_comp(sens_pupil_dilation, pupil_pos.data[:, tr_indexes])
