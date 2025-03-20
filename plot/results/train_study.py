import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import importlib.util

import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)

experiment_dir = 'train_test_001'

model_dir = os.path.join('data', 'experiments', experiment_dir)
plot_dir = os.path.join('plot', 'results', 'train', experiment_dir)
os.makedirs(plot_dir, exist_ok=True)

spec = importlib.util.spec_from_file_location('train_config.py', location=model_dir + '/train_config.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = spec

if module.folds == 'all':
    module.folds = np.arange(7)

os.makedirs(plot_dir, exist_ok=True)

for fold in module.folds:
    
    fold = 'fold_' + str(fold)
    model_dir_par = os.path.join(model_dir + '/' + fold, 'log.csv')
    if not os.path.exists(model_dir_par): break
    df = pd.read_csv(model_dir_par, skiprows=[4])
    epoch = np.arange(len(df['val_loss']))
    plt.figure(1)
    plt.plot(epoch, df['val_corr'], label='Val corr ' + fold)
    plt.legend()
    plt.title('Val Correlation Evolution')
    plt.figure(2)
    plt.plot(epoch, df['lr'], label='lr evo')
    plt.legend()
    plt.title('Learning Rate Evolution')
    plt.figure(3)
    plt.plot(epoch, df['train_loss'], label='train loss ' + fold)
    plt.legend()
    plt.title('Training Loss Evolution')
plt.figure(1)
plt.savefig(os.path.join(plot_dir, 'val_corr_evo.png'))
plt.figure(2)
plt.savefig(os.path.join(plot_dir, 'lr_evo.png'))
plt.figure(3)
plt.savefig(os.path.join(plot_dir, 'training_loss_evo.png'))