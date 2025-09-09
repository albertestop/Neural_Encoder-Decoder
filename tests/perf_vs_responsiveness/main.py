import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

models = [
    'train_test_019',
    'train_test_018_1',
    'train_test_017_1',
    'train_test_015',
    'train_test_016'
]
reconstructions = [
    '141',
    '139',
    '136',
    '120',
    '123'
]


responsivenes = []
enc_perf = []
for model in models:
    file_path = '/home/albertestop/Sensorium/data/experiments/' + model + '/fold_0/responsiveness/neural_response_study.txt'
    with open(file_path, "r") as f:
        lines = f.readlines()
    model_responsiveness = []
    for i in [2, 3, 4]:
        parts = lines[i].split()
        last_num = float(parts[-1])
        model_responsiveness.append(last_num)
    responsivenes.append(model_responsiveness)
    enc_perf.append(float(lines[0].split('-')[-1]))
perc_10, perc_25, perc_40 = zip(*responsivenes)

dec_perf = []
for dec_run in reconstructions:
    dec_path = '/home/albertestop/Sensorium/Clopath/reconstructions/results/' + dec_run 
    mouse = [d for d in os.listdir(dec_path) if os.path.isdir(os.path.join(dec_path, d))][0]
    dec_path = dec_path + '/' + mouse + '/rec_perf.txt'
    with open(dec_path, "r") as f:
        lines = f.readlines()
    dec_perf.append(float(lines[0].split()[-1]))

current_directory = Path(__file__).resolve().parent

plt.xlabel('N of neurons')
plt.ylabel('Performance')
plt.title('Performance vs N of neurons with corr above 0.10')
plt.scatter(perc_10, enc_perf, label = 'Enc Perf')
plt.scatter(perc_10, dec_perf, label = 'Dec Perf')
plt.legend()
plt.savefig(current_directory / Path('10_perc.png'))
plt.close()

plt.cla()
plt.clf()
plt.xlabel('N of neurons')
plt.ylabel('Performance')
plt.title('Performance vs N of neurons with corr above 0.25')
plt.scatter(perc_25, enc_perf, label = 'Enc Perf')
plt.scatter(perc_25, dec_perf, label = 'Dec Perf')
plt.legend()
plt.savefig(current_directory / Path('25_perc.png'))
plt.close()

plt.cla()
plt.clf()
plt.xlabel('N of neurons')
plt.ylabel('Performance')
plt.title('Performance vs N of neurons with corr above 0.40')
plt.scatter(perc_40, enc_perf, label = 'Enc Perf')
plt.scatter(perc_40, dec_perf, label = 'Dec Perf')
plt.legend()
plt.savefig(current_directory / Path('40_perc.png'))
plt.close()