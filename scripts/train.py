import sys
from pathlib import Path
import shutil

# Añadir el directorio raíz del proyecto al PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import os
import time
import copy
import json
import argparse
import gc
from pprint import pprint
from importlib.machinery import SourceFileLoader
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from argus import load_model
from argus.callbacks import (
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR,
    Checkpoint,
    LambdaLR,
    Callback,
)

# Importar las clases desde datasets_without_animal.py
from configs import train_config
from src.datasets import (
    TrainMouseVideoDataset,
    ValMouseVideoDataset,
    ConcatMiceVideoDataset,
)
from src.utils import get_lr, init_weights, get_best_model_path
from src.responses import get_responses_processor
from src.ema import ModelEma, EmaCheckpoint
from src.inputs import get_inputs_processor
from src.metrics import CorrelationMetric
from src.indexes import IndexesGenerator
from src.argus_models import MouseModel
from src.data import get_mouse_data
from src.mixers import CutMix
from src import constants

# Configurar PYTORCH_CUDA_ALLOC_CONF al inicio del script
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.backends.cudnn.benchmark = True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True, type=str)
    parser.add_argument("-f", "--folds", default="all", type=str)
    return parser.parse_args()

def print_detailed_gpu_memory():
    print("\nUso detallado de memoria GPU:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Memoria total: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Memoria asignada: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Memoria reservada: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
        print(f"  Memoria máxima asignada: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")

def aggressive_memory_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("Limpieza de memoria agresiva realizada.")
    print_detailed_gpu_memory()

# Definir el callback personalizado para actualizar el EMA
class EmaUpdateCallback(Callback):
    def __init__(self):
        super().__init__()

    def batch_complete(self, state):
        model = state.model
        model_ema = model.module.model_ema
        nn_module = model.module.nn_module

        if model_ema is not None:
            model_ema.update(nn_module)

def train_mouse(train_config, save_dir: Path, train_splits: list[str], val_splits: list[str]):
    config = copy.deepcopy(train_config.config)
    argus_params = config["argus_params"]

    # Seleccionar solo el ratón indicado por mouse_index
    mouse_indices = train_config.mouse_indices
    mice_to_use = [constants.mice[i] for i in mouse_indices]
    num_neurons_to_use = [constants.num_neurons[i] for i in mouse_indices]  # Mantener como lista
    num_mice_used = len(mouse_indices)

    # Actualizar readout_outputs en argus_params
    argus_params['nn_module'][1]['readout_outputs'] = num_neurons_to_use  # Usar la lista

    print("Creando modelo...")
    model = MouseModel(argus_params)
    print_detailed_gpu_memory()

    print("Aplicando DataParallel...")
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    print_detailed_gpu_memory()

    if config.get("init_weights", False):
        print("Iniciando inicialización de pesos...")
        init_weights(model.module.nn_module)
        print_detailed_gpu_memory()

    if config.get("ema_decay", False):
        print(f"Configurando EMA decay: {config['ema_decay']}")
        model.module.model_ema = ModelEma(model.module.nn_module, decay=config["ema_decay"])
        checkpoint_class = EmaCheckpoint
    else:
        checkpoint_class = Checkpoint

    if "distill" in config:
        print("Configurando modelo de destilación...")
        distill_params = config["distill"]
        distill_experiment_dir = constants.experiments_dir / distill_params["experiment"] / val_splits[0]
        distill_model_path = get_best_model_path(distill_experiment_dir)
        distill_model = load_model(distill_model_path, device=argus_params["device"])
        distill_model = nn.DataParallel(distill_model)
        distill_model.eval()
        distill_nn_module = distill_model.module.nn_module
        model.module.distill_model = distill_nn_module
        model.module.distill_ratio = distill_params["ratio"]
        print(f"Modelo de destilación cargado: {str(distill_model_path)}, ratio: {distill_params['ratio']}")
        print_detailed_gpu_memory()

    print("Configurando generadores y procesadores...")
    indexes_generator = IndexesGenerator(**argus_params["frame_stack"])
    inputs_processor = get_inputs_processor(*argus_params["inputs_processor"])
    responses_processor = get_responses_processor(*argus_params["responses_processor"])

    cutmix = CutMix(**config["cutmix"])

    print("Creando datasets de entrenamiento...")
    train_datasets = []
    mouse_epoch_size = config["train_epoch_size"] // len(train_config.mouse_indices)
    for mouse in mice_to_use:
        train_datasets.append(
            TrainMouseVideoDataset(
                load_params=train_config.data_load,
                mouse_data=get_mouse_data(mouse=mouse, splits=train_splits),
                indexes_generator=indexes_generator,
                inputs_processor=inputs_processor,
                responses_processor=responses_processor,
                epoch_size=mouse_epoch_size,
                mixer=cutmix,
            )
        )
    # Pasar mice_indexes al crear el dataset concatenado
    train_dataset = ConcatMiceVideoDataset(train_datasets)
    print(f"Tamaño del dataset de entrenamiento: {len(train_dataset)}")
    print_detailed_gpu_memory()

    print("Creando datasets de validación...")
    val_datasets = []
    for mouse in mice_to_use:
        val_datasets.append(
            ValMouseVideoDataset(
                load_params=train_config.data_load,
                mouse_data=get_mouse_data(mouse=mouse, splits=val_splits),
                indexes_generator=indexes_generator,
                inputs_processor=inputs_processor,
                responses_processor=responses_processor,
            )
        )
    val_dataset = ConcatMiceVideoDataset(val_datasets)
    print(f"Tamaño del dataset de validación: {len(val_dataset)}")
    print_detailed_gpu_memory()

    print("Creando DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_dataloader_workers"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"] // argus_params["iter_size"],
        num_workers=config["num_dataloader_workers"],
        shuffle=False,
    )
    print("DataLoaders creados.")
    print_detailed_gpu_memory()

    for num_epochs, stage in zip(config["num_epochs"], config["stages"]):
        print(f"\nIniciando stage: {stage} por {num_epochs} épocas")
        callbacks = [
            LoggingToFile(save_dir / "log.txt", append=True),
            LoggingToCSV(save_dir / "log.csv", append=True),
            EmaUpdateCallback(),  # Agregar el callback para actualizar el EMA
        ]

        num_iterations = (len(train_dataset) // config["batch_size"]) * num_epochs
        if stage == "warmup":
            print("Configurando LambdaLR para warmup")
            callbacks += [
                LambdaLR(lambda x: x / num_iterations,
                         step_on_iteration=True),
            ]
        elif stage == "train":
            print("Configurando checkpoint y CosineAnnealingLR")
            checkpoint_format = "model-{epoch:03d}-{val_corr:.6f}.pth"
            callbacks += [
                checkpoint_class(save_dir, file_format=checkpoint_format, max_saves=1),
                CosineAnnealingLR(
                    T_max=num_iterations,
                    eta_min=get_lr(config["min_base_lr"], config["batch_size"]),
                    step_on_iteration=True,
                ),
            ]

        metrics = [
            CorrelationMetric(),
        ]

        print("Iniciando entrenamiento...")
        model.module.fit(
            train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            callbacks=callbacks,
            metrics=metrics
        )
        print("Entrenamiento completado.")

if __name__ == "__main__":
    # Añadir el directorio raíz del proyecto al PYTHONPATH (nuevamente por si acaso)
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))

    if train_config.folds == "all":
        folds_splits = constants.folds_splits
    else:
        folds_splits = [f"fold_{fold}" for fold in train_config.folds.split(",")]

    experiment_dir = constants.experiments_dir / train_config.experiment
    print(f"Directorio del experimento: {experiment_dir}")
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)
        print("Directorio del experimento creado.")
    else:
        print(f"El directorio del experimento ya existe.")

    # Copiar el script actual al directorio del experimento
    script_path = Path(__file__)
    with open(experiment_dir / "train.py", "w") as outfile:
        outfile.write(script_path.read_text())
    print("Archivo train.py actualizado en el directorio del experimento.")

    # Guardar la configuración en un archivo JSON
    shutil.copy('configs/train_config.py', experiment_dir)
    print("Archivo config copiado en el directorio del experimento.")

    for fold_split in folds_splits:

        fold_experiment_dir = experiment_dir / fold_split

        val_folds_splits = [fold_split]
        train_folds_splits = sorted(set(constants.folds_splits) - set(val_folds_splits))

        print(f"\nFold de validación: {val_folds_splits}")
        print(f"Folds de entrenamiento: {train_folds_splits}")
        print(f"Directorio del experimento para este fold: {fold_experiment_dir}")

        # Crear el directorio del fold si no existe
        if not fold_experiment_dir.exists():
            fold_experiment_dir.mkdir(parents=True, exist_ok=True)
            print(f"Directorio del fold creado.")
        else:
            print(f"El directorio del fold ya existe.")

        # Llamar a train_mouse con el índice del ratón
        train_mouse(train_config, fold_experiment_dir, train_folds_splits, val_folds_splits)

    print("Entrenamiento completado para todos los ratones.")
    print("Experimento completado.")
