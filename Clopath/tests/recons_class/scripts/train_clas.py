from pathlib import Path
import sys
import os
import numpy as np

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent.parent.parent
sys.path.append(str(parent_dir))

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from Clopath.tests.recons_class.src.model import ReconsClassModel
from Clopath.tests.recons_class.src.generate_instances import generate_instances, split_dataset
from Clopath.tests.recons_class.src.training import training_loop
from Clopath.tests.recons_class.src.dataset import ReconsDataset
from Clopath.tests.recons_class.src.performance import performance


dataset_name = '1'
device = 'cuda'
dropout = 0.25

print('Starting execution')

print('\nLoading data...')

dataset = np.load(Path.cwd() / 'Clopath' / 'tests' / 'recons_class' / 'datasets' / dataset_name / 'dataset.npy')

print('Data loaded')

print('\nGenerating training datasets...')

instances, targets = generate_instances(dataset)
train_instances, train_targets, val_instances, val_targets = split_dataset(instances, targets)

train_data = ReconsDataset(train_instances, train_targets)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_data = ReconsDataset(val_instances, val_targets)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
test_loader = DataLoader(val_data, batch_size=1, shuffle=True)

print('Training len: ' + str(len(train_data)))
print('Validation len: ' + str(len(val_data)))
print('Datasets ready')
print('\nStarting Training...')

model = ReconsClassModel(train_instances[0].shape[0])
model.to(device=device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
loss_fn = nn.BCEWithLogitsLoss()   #nn.MSELoss()

model = training_loop(
            n_epochs = 100,
            device = device,
            optimizer = optimizer,
            scheduler = scheduler,
            model = model,
            loss_fn = loss_fn,   
            train_loader = train_loader,
            val_loader=val_loader,
            test_loader=test_loader
        )

performance(model, test_loader, device)

save_bool = input('Save model? (Y/N) ')
if save_bool == 'Y':
    torch.save(model.state_dict(), 
    'Clopath/tests/recons_class/models/clas_model.pth')