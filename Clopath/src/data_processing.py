import torch
import numpy as np
from src.inputs import get_inputs_processor
from src.responses import get_responses_processor
from Clopath.src_pipeline.responses_utils import *
from Clopath.src_pipeline.pupil_center_utils import *
from Clopath.src_pipeline.behavior_utils import *
from Clopath.src_pipeline.video_utils import *


class Pipeline:
    def __init__(self, trial_index, device):
        self.trial_index = trial_index
        self.device = device
        self.pc_pipeline = PupilCenterPipeline()
        self.responses_pipeline = ResponsesPipeline(trial_index, device)
        self.video_pipeline = VideoPipeline()
        self.behavior_pipeline = BehaviorPipeline()


    def __call__(self, model, video, behavior, pupil_center, responses, proc_params):
        video_params = proc_params['video_params']
        responses_params = proc_params['responses_params']
        pc_params = proc_params['pc_params']
        behavior_params = proc_params['behavior_params']

        video = self.video_pipeline(video, video_params)
        
        responses, population_mask = self.responses_pipeline(responses, responses_params)

        pupil_center = self.pc_pipeline(pupil_center, pc_params)

        behavior = self.behavior_pipeline(behavior, behavior_params)

        inputs_processor = get_inputs_processor(*model.params["inputs_processor"])
        inputs = inputs_processor(video, behavior, pupil_center)
        responses_processor = get_responses_processor(*model.params["responses_processor"])
        responses = responses_processor(responses)

        return inputs, video, behavior, pupil_center, responses, population_mask