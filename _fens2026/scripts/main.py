from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from _fens2026.modules.recons_study import compute_recons_metrics
from _fens2026.modules.reconstruct_session import recons_session

session = '2025-07-04_04_ESPM154'
run = '1'
metric_window_t = 1
r_type = 'sleep'
reconstruct = True

if reconstruct: recons_session(r_type, session)
compute_recons_metrics(session, metric_window_t)