from pathlib import Path

import numpy as np

import torch
import argus

from src.indexes import IndexesGenerator
from src.inputs import get_inputs_processor
from src.argus_models import MouseModel
from src import constants


def get_blend_weights(name: str, size: int):
    if name == "ones":
        return np.ones(size, dtype=np.float32)
    elif name == "linear":
        return np.linspace(0, 1, num=size)
    else:
        raise ValueError(f"Blend weights '{name}' is not supported")


class Predictor:
    def __init__(self, model_path: Path | str, device: str = "cuda:0", blend_weights="ones"):
        self.model: MouseModel = argus.load_model(model_path, device=device, optimizer=None, loss=None)
        self.model.eval()
        self.inputs_processor = get_inputs_processor(*self.model.params["inputs_processor"])
        self.frame_stack_size = self.model.params["frame_stack"]["size"]
        self.frame_stack_step = self.model.params["frame_stack"]["step"]
        assert self.model.params["frame_stack"]["position"] == "last"
        assert self.model.params["responses_processor"][0] == "identity"
        self.indexes_generator = IndexesGenerator(self.frame_stack_size,
                                                  self.frame_stack_step)
        self.blend_weights = get_blend_weights(blend_weights, self.frame_stack_size)

    @torch.no_grad()
    def predict_trial(self,
                      video: np.ndarray,
                      behavior: np.ndarray,
                      pupil_center: np.ndarray,
                      mouse_index: int) -> np.ndarray:
        inputs = self.inputs_processor(video, behavior, pupil_center).to(self.model.device)
        length = video.shape[-1]
        responses = np.zeros((constants.num_neurons[mouse_index], length), dtype=np.float32)
        blend_weights = np.zeros(length, np.float32)
        for index in range(
            self.indexes_generator.behind,
            length - self.indexes_generator.ahead
        ):
            indexes = self.indexes_generator.make_indexes(index)
            prediction = self.model.predict(inputs[:, indexes].unsqueeze(0), mouse_index)[0]
            responses[..., indexes] += prediction.cpu().numpy()
            blend_weights[indexes] += self.blend_weights
        responses /= np.clip(blend_weights, 1.0, None)
        return responses


def generate_predictors(device, model_path, number_models, model_list):
    """
    Generates a predictor (see src.predictors) for each neural response prediction model

    Seems to be wrong:
    if optimize_given_predicted_responses = True: we will generate a predictor
    for each model in model_path
    if optimize_given_predicted_responses = False: we will generate a predictor
    for each model in model_path[model_list]
    If optimize we should just generate predictors for used models,
    not all of them, but it is coded the other way round

    IN AN OLD VERSION OF THE MODEL, IF OPT_FOR_PRED_RESPONSES = True, WE WOULD GENERATE 7 
    PREDICTORS FOR THE 7 MODELS IN CONFIG, EVEN IF WE SPECIFIED THAT WE ARE USING LESS MODELS
    """
    predictor = [None] * number_models
    for n in range(len(predictor)):
        model_path_temp = model_path[model_list[n]]
        print('Predicting true responses with model: ', model_path_temp)
        predictor[n] = Predictor(model_path=model_path_temp, device=device, blend_weights="ones")
    return predictor