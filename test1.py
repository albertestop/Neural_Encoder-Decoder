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
import rastermap
import shutil
import sys

import argus
from argus import load_model
from torch import nn
from src.argus_models import MouseModel
from src.data import save_fold_tiers
from configs.train_config import config, data_load
import pathlib
import importlib

model_params_path = 'data.experiments.train_test_010.train_config'
config = importlib.import_module(model_params_path)
print(config.folds)