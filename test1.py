import torch
import os
import copy
import numpy as np
from PIL import Image
import pickle
import json
import matplotlib.pyplot as plt
import pandas as pd
from src import constants
from scipy.signal import find_peaks


import argus
from argus import load_model
from torch import nn
from src.argus_models import MouseModel
from configs.train_config import config, data_load
from configs.data_proc_001 import data

import shutil
import numpy as np
import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)
from proc_resources import response_proc, video_proc, behavior_proc, pupil_pos_proc

# lab_data_dir = '/home/albertestop/data/processed_data/sensorium_all_2023/2025-03-05_02_ESMT204_001'
# new_data_dir = '/home/albertestop/data/processed_data/sensorium_all_2023/2025-03-05_02_ESMT204_002'


# shutil.copytree(lab_data_dir, new_data_dir, dirs_exist_ok=True)
# for trial in os.listdir(os.path.join(lab_data_dir, 'data', 'responses')):
#     lab_response = np.load(os.path.join(lab_data_dir, 'data', 'responses', trial))
#     duplicated_arr = np.repeat(lab_response, 1000, axis=0)
#     np.save(os.path.join(new_data_dir, 'data', 'responses', trial), duplicated_arr)

# mouse_dir = os.path.join('/home/albertestop/data/processed_data/sensorium_all_2023', 'dynamic29515')
# responses = np.load(os.path.join('/home/albertestop/data/processed_data/sensorium_all_2023', 'dynamic29515', "data", "responses", "0.npy"))
#  #tiers_sens = np.load(str(constants.sensorium_dir / constants.new_mice[0] / "meta" / "trials" / "tiers.npy"))

# #  # Create tiers.py
# files = [f for f in os.listdir(os.path.join(mouse_dir, 'data', 'videos')) if os.path.isfile(os.path.join(os.path.join(mouse_dir, 'data', 'videos'), f))]
# j = len(files)
# count_final_test_bonus = int(j * 0.19)
# count_final_test_main = int(j * 0.08)
# count_live_test_bonus = int(j * 0.08)
# count_live_test_main = int(j * 0.08)
# count_oracle = int(j * 0.08)
# count_train = j - (count_final_test_bonus + count_final_test_main +
#                    count_live_test_bonus + count_live_test_main +
#                    count_oracle)
# elements = (['final_test_bonus'] * count_final_test_bonus +
#             ['final_test_main']  * count_final_test_main +
#             ['live_test_bonus']  * count_live_test_bonus +
#             ['live_test_main']   * count_live_test_main +
#             ['oracle']           * count_oracle +
#             ['train']            * count_train)
# np.random.shuffle(elements)
# tiers = np.array(elements)
# os.makedirs(os.path.join(mouse_dir, 'meta/trials/'), exist_ok=True)
# np.save(os.path.join(mouse_dir, 'meta/trials/tiers.npy'), tiers)
# print('Tiers file created.')
# # # Create cell_motor_coordinates.py
# cell_motor_coords = np.zeros((len(responses[:, 0]), 3))
# os.makedirs(os.path.join(mouse_dir, 'meta/neurons/'), exist_ok=True)
# np.save(os.path.join(mouse_dir, 'meta/neurons/cell_motor_coordinates.npy'), cell_motor_coords)
# print('Neuron coords file created.')
# # Create unit_ids.py
# unit_ids = np.arange(1, len(responses[:, 0]) + 1)
# np.save(os.path.join(mouse_dir, 'meta/neurons/unit_ids.npy'), unit_ids)
# print('Unit ids file created.')


tiers_lab = np.load(os.path.join('/home/albertestop/data/processed_data/sensorium_all_2023', 'dynamic29515', "meta", "trials", "tiers.npy"))
ids_lab = np.load(os.path.join('/home/albertestop/data/processed_data/sensorium_all_2023', 'dynamic29515', "meta", "neurons", "unit_ids.npy"))
coords_lab = np.load(os.path.join('/home/albertestop/data/processed_data/sensorium_all_2023', 'dynamic29515', "meta", "neurons", "cell_motor_coordinates.npy"))

tiers_sens = np.load(os.path.join('/home/albertestop/data/processed_data/sensorium_all_2023', 'dynamic29515', "meta", "trials", "temp", "tiers.npy"))
ids_sens = np.load(os.path.join('/home/albertestop/data/processed_data/sensorium_all_2023', 'dynamic29515', "meta", "neurons", "temp", "unit_ids.npy"))
coords_sens = np.load(os.path.join('/home/albertestop/data/processed_data/sensorium_all_2023', 'dynamic29515', "meta", "neurons", "temp", "cell_motor_coordinates.npy"))

print('\nTIERS')

tiers_lab_vals, counts = np.unique(tiers_lab, return_counts=True)
tiers_sens_vals, counts_sens = np.unique(tiers_sens, return_counts=True)

print(tiers_lab.shape, type(tiers_lab[0]), tiers_sens.shape, type(tiers_sens[0]))
print(tiers_lab_vals, counts)
print(tiers_sens_vals, counts_sens)

print('\nCOORDS')

print(coords_lab.shape, type(coords_lab[0, 0]), coords_sens.shape, type(coords_sens[0, 0]))
print(np.min(coords_lab), np.max(coords_lab))
print(np.min(coords_sens), np.max(coords_sens))

print('\nIDS')

print(ids_lab.shape, type(ids_lab[0]), ids_sens.shape, type(ids_sens[0]))
print(np.min(ids_lab), np.max(ids_lab), np.min(ids_sens), np.max(ids_sens))