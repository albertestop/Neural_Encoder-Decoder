from pathlib import Path
import pandas as pd

home_dir = Path.home()
work_dir = home_dir / "Sensorium"
data_dir = work_dir / "data"
sensorium_dir = home_dir / "data" / "processed_data" / "sensorium_all_2023"

configs_dir = work_dir / "configs"
experiments_dir = data_dir / "experiments"
predictions_dir = data_dir / "predictions"

new_mice = [
    "2025-07-07_05_ESPM154_002",
]
df_data = pd.read_csv(sensorium_dir / 'datasets.csv')
new_num_neurons = df_data[df_data['mouse'].isin(new_mice)].set_index('mouse').loc[new_mice]['n_neurons'].tolist()
old_mice = [
    # "dynamic29156",
    # "dynamic29228",
    # "dynamic29234",
    # "dynamic29513",
    # "dynamic29514",
]
old_num_neurons = []

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
