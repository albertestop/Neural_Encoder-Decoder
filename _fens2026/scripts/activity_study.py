from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from _fens2026.src.activity.metrics_comp import compute_activity_metrics

sessions = [
    '2025-07-04_04_ESPM154_008_recons'
    ]
reconstructions = [
    '0'
    ]
metric_window_t = 1

for session, reconstruction in zip(sessions, reconstructions):
    print(session)

    compute_activity_metrics(session, reconstruction, metric_window_t)