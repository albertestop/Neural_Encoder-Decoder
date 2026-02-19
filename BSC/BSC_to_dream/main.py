import subprocess
import sys
import shutil
from pathlib import Path

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))
sys.path.append(str(Path.home()))

from BSC.reconstruct.src import *
from my_utils.data_loading import *
from my_utils.dir_management import *

process_0 = 0
n_processes = 8

save_path = "/home/albertestop/Sensorium/BSC/BSC_to_dream/temp"
for item in os.listdir(save_path):
    if item == '.gitignore': continue
    item_path = os.path.join(save_path, item)
    if os.path.isfile(item_path) or os.path.islink(item_path):
        os.remove(item_path)
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

for i in range(n_processes):
    print("Transfering Sensorium" + str(process_0 + i))
    process_n = process_0 + i
    process_name = 'Sensorium' + str(process_n)
    process_path = "uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/" + process_name + "/Clopath/reconstructions/results/1"
    """cp_folds0 = "scp -r " + process_path + " " + save_path
    try:
        subprocess.run(cp_folds0, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Return code:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)"""
    print(f'rm -r "/gpfs/projects/uab103/uab020077/{process_name}/Clopath/reconstructions/results/"')
    print('deleting directory')
    try:
        enc_run = ["ssh", "uab020077@alogin1.bsc.es", f'rm -r "/gpfs/projects/uab103/uab020077/{process_name}/Clopath/reconstructions/results/1/"']
        subprocess.run(enc_run, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("SCP failed:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)

    folders = [f for f in os.listdir(save_path + '/1') if os.path.isdir(os.path.join(save_path + '/1', f))]
    print(folders)
    if len(folders) > 1: ValueError("More than one folder in temp")
    else: session = folders[0]
    source_dir = save_path + "/1/" + session
    if i == 0:
        proc_config_path = "/home/albertestop/data/processed_data/sensorium_all_2023/" + session + "/config.py"
        proc_config = load_config(proc_config_path)
        target_dir = proc_config.exp_directory + proc_config.animal + '/' + proc_config.session + '/reconstructions'
        os.makedirs(target_dir, exist_ok=True)
        target_dir =  next_num_folder(target_dir) + '/reconstruction'
        os.makedirs(target_dir, exist_ok=True)

    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)
        shutil.move(source_path, target_path)
    
