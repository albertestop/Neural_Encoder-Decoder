import subprocess

from BSC.reconstruct.src import *

session = ""
BSC_subpath_n = [
    "",
]

save_n = next_folder_number("/home/albertestop/Sensorium/Clopath/reconstructions/results/from_BSC")
save_path = "/home/albertestop/Sensorium/Clopath/reconstructions/results/from_BSC" + "/" + str(save_n)
os.mkdir(save_path)


for subpath in BSC_subpath_n:
    server_path = "uab020077@transfer1.bsc.es:/gpfs/projects/uab103/uab020077/Sensorium" + subpath + "/Clopath/reconstructions/results/0/" + session
    cp_folds0 = "scp -r " + server_path + " " + save_path
    try:
        subprocess.run(cp_folds0, shell=True, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Return code:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
    
    try:
        enc_run = ["ssh", "uab020077@alogin1.bsc.es", f'rm -r "/gpfs/projects/uab103/uab020077/Sensorium' + subpath + '/Clopath/reconstructions/results/0"']
        subprocess.run(enc_run, sell=True, capture_output=True, text=True, check=True)

    except subprocess.CalledProcessError as e:
        print("SCP failed:", e.returncode)
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)