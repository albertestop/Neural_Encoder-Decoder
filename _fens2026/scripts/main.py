from pathlib import Path
import sys

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from _fens2026.modules.recons_study import compute_recons_metrics
from _fens2026.modules.reconstruct_session import recons_session

sessions = [
    '2025-04-01_02_ESPM127_013_sleep',
    '2025-04-01_01_ESPM127_014_recons_random_all',
    '2025-04-01_01_ESPM127_014_recons_random_time',
    '2025-04-01_01_ESPM127_014_recons_random_neurons',
    '2025-04-01_02_ESPM127_013_sleep_random_all',
    '2025-04-01_02_ESPM127_013_sleep_random_time',
    '2025-04-01_02_ESPM127_013_sleep_random_neurons'
    ]
reconstructions = [
    '0',
    '1',
    '2',
    '3',
    '1',
    '2',
    '3'
    ]
metric_window_t = 1
build_session_recons = True
build_session_projections = False
compute_metrics = True

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
