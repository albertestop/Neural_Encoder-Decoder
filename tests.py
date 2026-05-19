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
from PIL import Image

import subprocess
import pandas as pd

try:
    cp_folds0 = "scp -r /home/albertestop/visual_cortex_study/transformer_arch uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/transformer_arch/transformer_arch_0"
    subprocess.run(cp_folds0, shell=True, capture_output=True, text=True, check=True)
except subprocess.CalledProcessError as e:
    print("SCP failed:", e.returncode)
    print("STDOUT:\n", e.stdout)
    print("STDERR:\n", e.stderr)

