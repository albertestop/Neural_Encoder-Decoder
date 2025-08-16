import numpy as np
from collections import defaultdict
import json
from pathlib import Path

from src.phash import calculate_video_phash
from src.utils import get_length_without_nan
from src import constants


def create_videos_phashes(mouse: str) -> np.ndarray:
    mouse_dir = constants.sensorium_dir / mouse
    tiers = np.load(str(mouse_dir / "meta" / "trials" / "tiers.npy"))
    phashes = np.zeros(tiers.shape[0], dtype=np.uint64)
    for trial_id, tier in enumerate(tiers):
        if tier == "none":
            continue
        video = np.load(str(mouse_dir / "data" / "videos" / f"{trial_id}.npy"))
        phashes[trial_id] = calculate_video_phash(video)
    return phashes


def get_folds_tiers(mouse: str, num_folds: int):
    tiers = np.load(str(constants.sensorium_dir / mouse / "meta" / "trials" / "tiers.npy"))
    tiers = tiers.astype(object)  # Convert to object dtype to allow longer strings.
    phashes = create_videos_phashes(mouse)
    if mouse in constants.new_mice:
        trial_ids = np.argwhere((tiers == "train") | (tiers == "oracle")).ravel()
    else:
        trial_ids = np.argwhere(tiers != "none").ravel()
    for trial_id in trial_ids:
        fold = int(phashes[trial_id]) % num_folds  # group k-fold by video hash
        tiers[trial_id] = f"fold_{fold}"
    return tiers


def get_mouse_data(mouse: str, splits: list[str]) -> dict:
    assert mouse in constants.mice
    tiers = get_folds_tiers(mouse, constants.num_folds)
    mouse_dir = constants.sensorium_dir / mouse
    neuron_ids = np.load(str(mouse_dir / "meta" / "neurons" / "unit_ids.npy"))
    cell_motor_coords = np.load(str(mouse_dir / "meta" / "neurons" / "cell_motor_coordinates.npy"))

    mouse_data = {
        "mouse": mouse,
        "splits": splits,
        "neuron_ids": neuron_ids,
        "num_neurons": neuron_ids.shape[0],
        "cell_motor_coordinates": cell_motor_coords,
        "trials": [],
    }

    for split in splits:
        if split in constants.folds_splits:
            labeled_split = True
        elif split in constants.unlabeled_splits:
            labeled_split = False
        else:
            raise ValueError(f"Unknown data split '{split}'")
        trial_ids = np.argwhere(tiers == split).ravel().tolist()

        for trial_id in trial_ids:
            behavior_path = str(mouse_dir / "data" / "behavior" / f"{trial_id}.npy")
            trial_data = {
                "trial_id": trial_id,
                "length": get_length_without_nan(np.load(behavior_path)[0]),
                "video_path": str(mouse_dir / "data" / "videos" / f"{trial_id}.npy"),
                "behavior_path": behavior_path,
                "pupil_center_path": str(mouse_dir / "data" / "pupil_center" / f"{trial_id}.npy"),
            }
            if labeled_split:
                response_path = str(mouse_dir / "data" / "responses" / f"{trial_id}.npy")
                trial_data["response_path"] = response_path
                trial_data["length"] = get_length_without_nan(np.load(response_path)[0])
            mouse_data["trials"].append(trial_data)

    return mouse_data


def save_fold_tiers(mouse: str):
    tiers = get_folds_tiers(mouse, constants.num_folds)
    tiers = np.array([s[-1:] for s in tiers])
    grouped_folds = defaultdict(list)
    for trial_id, fold in enumerate(tiers):
        grouped_folds[fold].append(trial_id)
    grouped_folds = dict(grouped_folds)
    result_dict = {mouse: grouped_folds}
    with open(str(Path.home()) + "/Sensorium/Clopath/folds_trials.json", "r") as f:
        saved_data = json.load(f)
    if mouse not in saved_data.keys():
        saved_data.update(result_dict)
        with open(str(Path.home()) + "/Sensorium/Clopath/folds_trials.json", "w") as f:
            json.dump(saved_data, f, indent=4)
        
        print('\nFold of each trial of the mouse added in Clopath/folds_trials.json')


