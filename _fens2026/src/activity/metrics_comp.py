from pathlib import Path
import os
import sys
import numpy as np

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))



def compute_activity_metrics(session, reconstruction, metric_window_t):
    """
    mean activity
    % of activity explained by PCA
    MIND (file:///C:/Users/alber/Documents/Feina/24-10%20Ins%20Neurociencies%20UAB/Journal%20Club/25-10-31,%20IDIBAPS.pdf), (Low, R. J., et al. Probing variability in a cognitive map using manifold inference from neural dynamics. bioRxiv (2018) doi:10.1101/418939)
    """
    pass