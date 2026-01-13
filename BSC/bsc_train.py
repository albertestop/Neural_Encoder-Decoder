import subprocess
import pandas as pd
from src import constants

BSC_subpath = 'Sensorium'   # Sensorium, Sensorium, Sensorium2, Sensorium3

try:
    cp_folds0 = "scp -r /home/albertestop/Sensorium/Clopath/folds_trials.json uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/Sensorium/Clopath"
    subprocess.run(cp_folds0, shell=True, capture_output=True, text=True, check=True)
    cp_folds1 = "scp -r /home/albertestop/Sensorium/Clopath/folds_trials.json uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/Sensorium1/Clopath"
    subprocess.run(cp_folds1, shell=True, capture_output=True, text=True, check=True)
    cp_folds2 = "scp -r /home/albertestop/Sensorium/Clopath/folds_trials.json uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/Sensorium2/Clopath"
    subprocess.run(cp_folds2, shell=True, capture_output=True, text=True, check=True)
    cp_folds3 = "scp -r /home/albertestop/Sensorium/Clopath/folds_trials.json uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/Sensorium3/Clopath"
    subprocess.run(cp_folds3, shell=True, capture_output=True, text=True, check=True)

    cp_datasets_csv = "scp -r /home/albertestop/data/processed_data/sensorium_all_2023/datasets.csv uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/data/processed_data/sensorium"
    subprocess.run(cp_datasets_csv, shell=True, capture_output=True, text=True, check=True)

    print("Data files in BSC updated correctly")

except subprocess.CalledProcessError as e:
    print("SCP failed:", e.returncode)
    print("STDOUT:\n", e.stdout)
    print("STDERR:\n", e.stderr)


for session in constants.mice:
    df = pd.read_csv('bsc_datasets.csv', header=None)
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
        df.to_csv('bsc_datasets.csv', header=False, index=False)
        print("Sessions uploaded to BSC correctly")


try:
    cp_config = "scp -r /home/albertestop/Sensorium/configs/config.py uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/" + BSC_subpath + "/configs"
    subprocess.run(cp_config, shell=True, capture_output=True, text=True, check=True)
    cp_constants = "scp -r /home/albertestop/Sensorium/src/constants.py uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/" + BSC_subpath + "/src"
    subprocess.run(cp_constants, shell=True, capture_output=True, text=True, check=True)

    print("Config and constants files in BSC updated correctly")

except subprocess.CalledProcessError as e:
    print("SCP failed:", e.returncode)
    print("STDOUT:\n", e.stdout)
    print("STDERR:\n", e.stderr)


try:
    enc_run = ["ssh", "uab020077@alogin1.bsc.es", f'cd "/gpfs/projects/uab103/uab020077/' + BSC_subpath + '/scripts" && sbatch train.sh']
    subprocess.run(enc_run, capture_output=True, text=True, check=True)

    print("Train script sent to queue correctly")

except subprocess.CalledProcessError as e:
    print("SCP failed:", e.returncode)
    print("STDOUT:\n", e.stdout)
    print("STDERR:\n", e.stderr)

