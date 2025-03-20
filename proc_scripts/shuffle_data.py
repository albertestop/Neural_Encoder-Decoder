import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)

import pandas as pd
import numpy as np

from src import constants
from configs.data_proc_001 import data
from proc_resources import response_proc, video_proc, behavior_proc, pupil_pos_proc

mouse_directory = '/home/antoniofernandez/sensorium_data/'
mice = [
    'dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce',
    'dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20',
    'dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20',
    'dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20',
    'dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20',
    'dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20'
    ]
mouse_new_directory = '/home/albertestop/Sensorium/sensorium/data/sensorium_all_2023/random_1/'

for mouse in mice:
    files = os.listdir(mouse_directory + mouse + '/data/responses')
    numbers = []
    for file in files:
        number = int(os.path.splitext(file)[0])
        numbers.append(number)
    highest_number = max(numbers)

    for i in range(highest_number):
        print(mouse, i)
        if os.path.exists(mouse_directory + mouse + '/data/responses/' + str(i) + '.npy'):
            response = np.load(mouse_directory + mouse + '/data/responses/' + str(i) + '.npy')
            np.random.shuffle(response)
            np.save(mouse_new_directory + mouse + '/data/responses/' + str(i) + '.npy', response)
            print(response.shape)