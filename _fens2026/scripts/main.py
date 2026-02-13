from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from _fens2026.modules.recons_study import compute_recons_metrics
from _fens2026.modules.reconstruct_session import recons_session

recons_run = 'from_BSC/0'
metric_window_t = 3
r_type = 'movie'
reconstruct = True

if reconstruct: recons_session(r_type, recons_run)
compute_recons_metrics(recons_run, metric_window_t)