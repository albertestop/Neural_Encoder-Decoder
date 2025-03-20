import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)

retrain = False
if retrain: num_folds = 7
else: num_folds = 1
retrain_periods = 1
model = 'true_batch_001/fold_0'
#model_2 = 'fold_0_without_last_mouse'

os.makedirs(os.path.join('plot', 'results', model), exist_ok=True)


if not retrain:
    experiment_dir = os.path.join(working_dir, 'data', 'experiments', model)
    #experiment_dir = os.path.join('/home/antoniofernandez/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/experiments/true_batch_001', model)
    #experiment_dir_2 = os.path.join('/home/antoniofernandez/code/Sensorium/sensorium-v23.11.22/lRomul-sensorium-6849050/data/experiments/true_batch_001', model_2)
else:
    experiment_dir = os.path.join(working_dir, 'data', 'experiments', model)

if retrain and retrain_periods == 2:

    folds_dir = os.path.join(experiment_dir, 'best_period_0', 'log.csv')
    df_0 = pd.read_csv(folds_dir)
    epoch = np.arange(len(df_0['val_loss']))
    plt.figure(1)
    plt.plot(epoch, df_0['val_corr'])
    plt.figure(3)
    plt.plot(epoch, df_0['train_loss'])


for fold in range(num_folds):
    fold = 'fold_' + str(fold)
    if retrain: 
        folds_dir = os.path.join(experiment_dir, fold, 'log.csv')
        df = pd.read_csv(folds_dir)
        model = fold
    else: 
        model_dir = os.path.join(experiment_dir, 'log.csv')
        #model_dir_2 = os.path.join(experiment_dir_2, 'log.csv')
        df = pd.read_csv(model_dir, skiprows=[4])
        #df_2 = pd.read_csv(model_dir_2, skiprows=[4])
    epoch = np.arange(len(df['val_loss']))
    if retrain and retrain_periods == 2: epoch += len(df_0)
    plt.figure(1)
    plt.plot(epoch, df['val_corr'], label='Val corr ' + model)
    #plt.plot(epoch, df_2['val_corr'], label='Val corr ' + model_2)
    plt.legend()
    plt.title('Val Correlation Evolution')
    plt.savefig(os.path.join('plot', 'results', model, 'val_corr_evo.png'))
    plt.figure(2)
    plt.plot(epoch, df['lr'], label='lr evo')
    #plt.plot(epoch, df_2['lr'], label='lr evo')
    plt.legend()
    plt.title('Learning Rate Evolution')
    plt.savefig(os.path.join('plot', 'results', model, 'lr_evo.png'))
    plt.figure(3)
    plt.plot(epoch, df['train_loss'], label='train loss ' + model)
    #plt.plot(epoch, df_2['train_loss'], label='train loss ' + model_2)
    plt.legend()
    plt.title('Training Loss Evolution')
    plt.savefig(os.path.join('plot', 'results', model, 'training_loss_evo.png'))