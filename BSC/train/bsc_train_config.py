import torch
from pathlib import Path

def get_lr(base_lr: float, batch_size: int, base_batch_size: int = 4) -> float:
    return base_lr * (batch_size / base_batch_size)

mouse_indices = [0]
folds = 'all'   # '0,1,2' (without space) // 'all'
experiment = Path('new_experiment')
iter_par = 2
image_size = (64, 64)
batch_size = 32
base_lr = 1.5e-4 * iter_par
frame_stack_size = 16
num_gpus = torch.cuda.device_count()
data_load = dict(
    videos_params = {
        'use_original': True,
        'regenerate': 'zeros',
    },
    response_params ={
        'use_original': True,
        'regenerate': 'zeros',
    },
    behavior_params ={
        'use_original': True,
        'regenerate_pupil_size': 'zeros',
        'regenerate_speed': 'zeros',
    },
    pupil_pos_params ={
        'use_original': True,
        'regenerate_position': 'zeros',
    },
)
config = dict(
    image_size=image_size,
    batch_size=batch_size,
    base_lr=base_lr,
    min_base_lr=base_lr * 0.01,
    ema_decay=0.999,
    train_epoch_size=8000,
    num_epochs=[3, 18],
    stages=["warmup", "train"],
    num_dataloader_workers=2 * num_gpus,  # Ajustamos el número de workers basado en las GPUs
    init_weights=True,
    argus_params={
        "nn_module": ("dwiseneuro", {
            "readout_outputs": 0,
            "in_channels": 5,
            "core_features": (64, 64, 64, 64,
                              128, 128, 128,
                              256, 256),
            "spatial_strides": (2, 1, 1, 1,
                                2, 1, 1,
                                2, 1),
            "spatial_kernel": 3,
            "temporal_kernel": 5,
            "expansion_ratio": 7,
            "se_reduce_ratio": 32,
            "cortex_features": (512 * 2, 1024 * 2, 2048 * 2),
            "groups": 2,
            "softplus_beta": 0.07,
            "drop_rate": 0.4,
            "drop_path_rate": 0.1,
        }),
        "loss": ("mice_poisson", {
            "log_input": False,
            "full": False,
            "eps": 1e-8,
        }),
        "optimizer": ("AdamW", {
            "lr": get_lr(base_lr, batch_size),
            "weight_decay": 0.05,
        }),
        "device": "cuda" if torch.cuda.is_available() else "cpu",  # DDP manejará la asignación específica del dispositivo
        "frame_stack": {
            "size": frame_stack_size,
            "step": 2,
            "position": "last",
        },
        "inputs_processor": ("stack_inputs", {
            "size": image_size,
            "pad_fill_value": 0.,
        }),
        "responses_processor": ("identity", {}),
        "amp": True,
        "iter_size": iter_par,
    },
    cutmix={
        "alpha": 1.0,
        "prob": 0.5,
    },
)
