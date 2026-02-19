import subprocess
import pandas as pd
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from reconstruct_config import *
from BSC.reconstruct.src import *

carry_on = input("Have you updated: \n- BSC_subpath variable in this file\n- mice variable in this file?\n- work_dir variable in bsc_constants?\n- mice in bsc_constants?\n- mask in bsc_config?\n- models in bsc_config?\n- trials in bsc_config?\n Y/N\n")
if carry_on != 'Y':
    exit()

BSC_subpath = 'Sensorium40'   # Sensorium, Sensorium, Sensorium2, Sensorium3
mice = [
    "2025-07-04_06_ESPM154_007_sleep_random_neurons",
]
process_0 = int(BSC_subpath[9:])
n_trials = end_trial - start_trial
n_models = len(user_model_list)
req_runtime = 7.5 * n_trials * n_models
n_processes = int(req_runtime / 3600) + 1
tr_i = start_trial
tr_f = end_trial

if (process_0 + n_processes + 1) > 47: 
    ValueError("Wait to execute, with this process we would exceed the max n of simultaneous processes in BSC (32)")

carry_on = input("Have you already saved prev reconstructions? Continuing will delete previous reconstruction data\nY/N\n")
if carry_on != 'Y':
    exit()
try:
    for i in range(32):
        cp_folds = "rm -rf uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/Sensorium" + str(int(i)) + "/Clopath/reconstructions/results/1"
        subprocess.run(cp_folds, shell=True, capture_output=True, text=True, check=True)

    print("Previous reconstructions deleted")

except subprocess.CalledProcessError as e:
    print("SCP failed:", e.returncode)
    print("STDOUT:\n", e.stdout)
    print("STDERR:\n", e.stderr)

print("Reconstructing a total of " + str(n_trials) + " trials, with " + str(n_models) + " models, what would require approx " + str(req_runtime) + " minutes (" + str(req_runtime / 60) + " hours)")
print("Using " + str(n_processes + 1) + " processes for the reconstruction, starting on Sensorium" + str(process_0) + ".")


try:
    for i in range(32):
        cp_folds = "scp -r /home/albertestop/Sensorium/Clopath/folds_trials.json uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/Sensorium" + str(int(i)) + "/Clopath"
        subprocess.run(cp_folds, shell=True, capture_output=True, text=True, check=True)

    cp_datasets_csv = "scp -r /home/albertestop/data/processed_data/sensorium_all_2023/datasets.csv uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/data/processed_data/sensorium"
    subprocess.run(cp_datasets_csv, shell=True, capture_output=True, text=True, check=True)

    print("Data files in BSC updated correctly")

except subprocess.CalledProcessError as e:
    print("SCP failed:", e.returncode)
    print("STDOUT:\n", e.stdout)
    print("STDERR:\n", e.stderr)


for session in mice:
    df = pd.read_csv('BSC/src/bsc_datasets.csv', header=None)
    datasets = df[0].dropna().astype(str).tolist()

    if session not in datasets:
        try:
            cp_dataset = "scp -r /home/albertestop/data/processed_data/sensorium_all_2023/" + session + " uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/data/processed_data/sensorium"
            subprocess.run(cp_dataset, shell=True, capture_output=True, text=True, check=True)

        except subprocess.CalledProcessError as e:
            print("SCP failed:", e.returncode)
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)

        df.loc[len(df)] = session
        df.to_csv('BSC/src/bsc_datasets.csv', header=False, index=False)
        print("Sessions uploaded to BSC correctly")


from BSC.reconstruct.reconstruct_config import pretrained_mask
df = pd.read_csv('BSC/src/bsc_masks.csv', header=None)
masks = df[0].dropna().astype(str).tolist()

if pretrained_mask not in masks:
    try:
        for i in range(32):
            cp_mask = "scp -r /home/albertestop/Sensorium/Clopath/reconstructions/masks/" + pretrained_mask + " uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/Sensorium" + str(int(i)) + "/Clopath/reconstructions/masks"
            subprocess.run(cp_mask, shell=True, capture_output=True, text=True, check=True)

    except subprocess.CalledProcessError as e:
        print("SCP failed:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)

    df.loc[len(df)] = pretrained_mask
    df.to_csv('BSC/src/bsc_masks.csv', header=False, index=False)
    print("Masks uploaded to BSC correctly")


for i in range(n_processes + 1):

    BSC_subpath_temp = str(process_0 + i)
    print("Running subpath " + BSC_subpath_temp)
    start_tr = i * int(n_trials / n_processes)
    end_tr = (i * int(n_trials / n_processes)) + int(n_trials / n_processes)

    update_py_constants(
        "/home/albertestop/Sensorium/BSC/reconstruct/reconstruct_config.py",
        {
            "start_trial": start_tr,
            "end_trial": end_tr,
        }
    )

    try:
        cp_config = "scp -r /home/albertestop/Sensorium/BSC/reconstruct/reconstruct_config.py uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/Sensorium" + BSC_subpath_temp + "/Clopath/scripts/config.py"
        subprocess.run(cp_config, shell=True, capture_output=True, text=True, check=True)
        cp_constants = "scp -r /home/albertestop/Sensorium/BSC/src/bsc_constants.py uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/Sensorium" + BSC_subpath_temp + "/src/constants.py"
        subprocess.run(cp_constants, shell=True, capture_output=True, text=True, check=True)

        print("Config and constants files in BSC updated correctly")

    except subprocess.CalledProcessError as e:
        print("SCP failed:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)


    try:
        enc_run = ["ssh", "uab020077@alogin1.bsc.es", f'cd "/gpfs/projects/uab103/uab020077/Sensorium' + BSC_subpath_temp + '/Clopath/scripts" && sbatch reconstruct.sh']
        subprocess.run(enc_run, shell=False, capture_output=True, text=True, check=True)

        print("Train script sent to queue correctly")

    except subprocess.CalledProcessError as e:
        print("SCP failed:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)

update_py_constants(
    "/home/albertestop/Sensorium/BSC/reconstruct/reconstruct_config.py",
    {
        "start_trial": tr_i,
        "end_trial": tr_f,
    }
)
