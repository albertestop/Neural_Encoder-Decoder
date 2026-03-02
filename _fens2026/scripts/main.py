from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from _fens2026.modules.recons_study import compute_recons_metrics
from _fens2026.modules.reconstruct_session import recons_session

sessions = [
    '2025-07-04_04_ESPM154_008_recons_random_all',
    '2025-07-04_04_ESPM154_008_recons_random_time',
    '2025-07-04_04_ESPM154_008_recons_random_neurons',
    ]
reconstructions = [
    '1',
    '2',
    '3',
    ]
metric_window_t = 1
build_session_recons = False
build_session_projections = True
compute_metrics = False

for session, reconstruction in zip(sessions, reconstructions):
    print(session)
    if build_session_recons or build_session_projections:
        recons_session(
            session,
            reconstruction,
            reconstruct_session=build_session_recons,
            recons_projections=build_session_projections,
        )

    if compute_metrics:
        compute_recons_metrics(session, reconstruction, metric_window_t)
