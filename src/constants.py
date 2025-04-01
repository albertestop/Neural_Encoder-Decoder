from pathlib import Path

home_dir = Path.home()
work_dir = home_dir / "Sensorium"
data_dir = work_dir / "data"
sensorium_dir = home_dir / "data" / "processed_data" / "sensorium_all_2023"

configs_dir = work_dir / "configs"
experiments_dir = data_dir / "experiments"
predictions_dir = data_dir / "predictions"

new_mice = [
    "2025-03-05_02_ESMT204_000",
    "dynamic29623",
    "dynamic29647",
    "dynamic29712",
    "dynamic29755",
]
new_num_neurons = [9, 7908, 8202, 7939, 8122]
old_mice = [
    "dynamic29156",
    "dynamic29228",
    "dynamic29234",
    "dynamic29513",
    "dynamic29514",
]
old_num_neurons = [7440, 7928, 8285, 7671, 7495]
"""
Mice:
2025-02-26_02_ESPM126_000, 2370
2025-03-05_02_ESMT204_000, 9
dynamic29515, 7863
"""

dataset2mice = {
    "new": new_mice,
    "old": old_mice,
}
mouse2dataset = {m: d for d, mc in dataset2mice.items() for m in mc}
# dataset2url_format = {
#     "new": "https://gin.g-node.org/pollytur/sensorium_2023_dataset/raw/master/{file_name}",
#     "old": "https://gin.g-node.org/pollytur/Sensorium2023Data/raw/master/{file_name}",
# }

mice = new_mice + old_mice
num_neurons = new_num_neurons + old_num_neurons

num_mice = len(mice)
index2mouse: dict[int, str] = {index: mouse for index, mouse in enumerate(mice)}
mouse2index: dict[str, int] = {mouse: index for index, mouse in enumerate(mice)}
mouse2num_neurons: dict[str, int] = {mouse: num for mouse, num in zip(mice, num_neurons)}
mice_indexes = list(range(num_mice))

unlabeled_splits = ["live_test_main", "live_test_bonus", "final_test_main", "final_test_bonus"]

num_folds = 7
folds = list(range(num_folds))
folds_splits = [f"fold_{fold}" for fold in folds]

submission_limit_length = 300
submission_skip_first = 50
submission_skip_last = 1
