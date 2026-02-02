import random
import numpy as np
import argus

import torch

from src.inputs import get_inputs_processor
from src.responses import get_responses_processor
from Clopath.src.data_processing import Pipeline
from src.predictors import generate_predictors

def load_trial_data(model, trial_index, responses, behavior, pupil_center, device, length, proc_params):

    video = np.ones((64, 64, length))
    video = video*(255/2)

    pipeline = Pipeline(trial_index, device)
    inputs, video, behavior, pupil_center, responses, population_mask = pipeline(model[0], video, behavior, pupil_center, responses, proc_params)

    inputs = inputs.to(device)
    responses = responses.to(device)
        
    return inputs, behavior, pupil_center, responses, population_mask