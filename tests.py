import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import json
import torch

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
from src.predictors import generate_predictors
from src.data import get_folds_tiers
from src.responsiveness import responsiveness
import random
import pickle
import imageio.v3 as iio
import subprocess
from src import constants



directory = Path("/home/albertestop/Sensorium/Clopath/reconstructions/masks")
npy_files = sorted(directory.rglob("*.npy"))   # use .glob("*.npy") for only top-level

for f in npy_files:
    print(str(f).split('/')[-1])