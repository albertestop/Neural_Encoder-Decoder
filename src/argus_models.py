import torch

import argus
from argus.engine import State
from argus.loss import pytorch_losses
from argus.utils import deep_to, deep_detach, deep_chunk

from src.ema import ModelEma
from src.losses import MicePoissonLoss
from src.models.dwiseneuro import DwiseNeuro


class MouseModel(argus.Model):
    nn_module = {
        "dwiseneuro": DwiseNeuro,
    }
    loss = {
        **pytorch_losses,
        "mice_poisson": MicePoissonLoss,
    }

    def __init__(self, params: dict):
        super().__init__(params)
        self.iter_size = int(params.get('iter_size', 1))
        self.amp = bool(params.get('amp', False))
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.model_ema: ModelEma | None = None
        self.distill_model: torch.nn.Module | None = None
        self.distill_ratio: float = 0.

    @torch.no_grad()
    def add_distill_predictions(self, input, target):
        if self.distill_model is not None and self.distill_ratio:
            distill_prediction = self.distill_model(input)
            target_tensors, mice_weights = target
            distill_mask = mice_weights == 0.
            distill_weight = (self.distill_ratio / (1. - self.distill_ratio)
                              * mice_weights.sum() / distill_mask.sum())
            for batch_idx, mouse_idx in torch.argwhere(distill_mask):
                target_tensors[mouse_idx][batch_idx] = distill_prediction[mouse_idx][batch_idx]
                mice_weights[batch_idx, mouse_idx] = distill_weight

    def train_step(self, batch, state: State) -> dict:
        self.train()
        self.optimizer.zero_grad()

        loss_value = 0
        for i, chunk_batch in enumerate(deep_chunk(batch, self.iter_size)):
            input, target = deep_to(chunk_batch, self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=self.amp):
                self.add_distill_predictions(input, target)
                prediction = self.nn_module(input)
                loss = self.loss(prediction, target)
                loss = loss / self.iter_size
            self.grad_scaler.scale(loss).backward()
            loss_value += loss.item()

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        if self.model_ema is not None:
            self.model_ema.update(self.nn_module)

        prediction = deep_detach(prediction)
        target = deep_detach(target)
        prediction = self.prediction_transform(prediction)
        return {
            'prediction': prediction,
            'target': target,
            'loss': loss_value
        }

    def val_step(self, batch, state: State) -> dict:
        self.eval()
        with torch.no_grad():
            input, target = deep_to(batch, device=self.device, non_blocking=True)
            if self.model_ema is None:
                prediction = self.nn_module(input)
            else:
                prediction = self.model_ema.ema(input)
            loss = self.loss(prediction, target)
            prediction = self.prediction_transform(prediction)
            return {
                'prediction': prediction,
                'target': target,
                'loss': loss.item()
            }

    def predict(self, input, mouse_index: int | None = None):
        self._check_predict_ready()
        with torch.no_grad():
            self.eval()
            input = deep_to(input, self.device)
            if self.model_ema is None:
                prediction = self.nn_module(input, mouse_index)
            else:
                prediction = self.model_ema.ema(input, mouse_index)
            prediction = self.prediction_transform(prediction)
            return prediction
