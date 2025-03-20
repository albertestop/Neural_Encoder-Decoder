import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)

sens_data_dir = '/home/albertestop/Sensorium/sensorium/data/sensorium_all_2023/dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20'
lab_data_dir = '/home/albertestop/Sensorium/sensorium/data/sensorium_all_2023/ESPM113_002'

trial = 0
correlations = []

while os.path.exists(os.path.join(lab_data_dir, 'data', 'responses', str(trial) + '.npy')):

    lab_response = np.load(os.path.join(lab_data_dir, 'data', 'responses', str(trial) + '.npy'))
    lab_response = np.nan_to_num(lab_response, nan=0)

    for i in range(len(lab_response[:, 0])):
        if not np.all(lab_response[i, :] == 0):
            if i > 0:
                if not np.all(lab_response[i - 1, :] == 0):
                    corr_above = abs(np.corrcoef(lab_response[i, :], lab_response[i - 1, :])[0, 1])
                    correlations.append(corr_above)
            if i < len(lab_response[:, 0]) - 1:
                if not np.all(lab_response[i + 1, :] == 0):
                    corr_below = abs(np.corrcoef(lab_response[i, :], lab_response[i + 1, :])[0, 1])
                    correlations.append(corr_below)
    
    trial +=1
    print(trial)


correlations = np.array(correlations)
lab_corr = np.mean(correlations)
print('Finished with lab correlations, number of trials analyzed = ' + str(trial) + '. Correlation = ' + str(lab_corr))

trial = 0
correlations = []

while os.path.exists(os.path.join(sens_data_dir, 'data', 'responses', str(trial) + '.npy')):

    sens_response = np.load(os.path.join(sens_data_dir, 'data', 'responses', str(trial) + '.npy'))
    sens_response = np.nan_to_num(sens_response, nan=0)


    for i in range(len(sens_response[:, 0])):
        if not np.all(sens_response[i, :] == 0):
            if i > 0:
                if not np.all(sens_response[i - 1, :] == 0):
                    corr_above = abs(np.corrcoef(sens_response[i, :], sens_response[i - 1, :])[0, 1])
                    correlations.append(corr_above)
            if i < len(sens_response[:, 0]) - 1:
                if not np.all(sens_response[i + 1, :] == 0):
                    corr_below = abs(np.corrcoef(sens_response[i, :], sens_response[i + 1, :])[0, 1])
                    correlations.append(corr_below)
    
    trial +=1
    print(trial)



correlations = np.array(correlations)
sens_corr = np.mean(correlations)
print('Finished with sensorium correlations, number of trials analyzed = ' + str(trial))


print('Mean correlation of sensorium responses: ' + str(sens_corr))
print('Mean correlation of lab responses: ' + str(lab_corr))

with open('plot/data/response_correlation/response_corr.txt' , "w") as file:
    file.write(f'Sens corr = ' + str(sens_corr) + '\n')
    file.write(f'Lab corr = ' + str(lab_corr) + '\n')
file.close()