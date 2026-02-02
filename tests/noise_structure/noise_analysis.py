import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import tifffile
from scipy.signal import welch
import shutil

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
sys.path.append(str(parent_dir))

from config import *
from Clopath.src.data_saving import *

def reconstruct_video(data, save_path, name):
    num_frames, width, height = data.shape

    tifffile.imwrite(save_path + '/output.tif', 
                    data.astype('uint8'),
                    imagej=True,
                    metadata = {'unit': 'um','fps': 30.0,'axes': 'TYX',})

    tif_frames = cv2.imreadmulti(save_path + '/output.tif')[1]

    height, width = tif_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter(save_path + '/' + name + str('.mp4'), fourcc, 30, (width, height))

    for frame in tif_frames:
        video_output.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    video_output.release()


save_path = generate_folder(str(current_dir) + save_folder)


recons_video = np.load(str(parent_dir) + '/' + recons_path)[0][0][:, 14:50, :].transpose(0, 2, 1)
gt_video = np.load(gt_path)[:, :, :300].transpose(2, 1, 0)

mask = np.load(parent_dir / Path(f'Clopath/reconstructions/masks/mask_m{mouse_index}.npy'))[:, 14:50]
mask = np.tile(mask,(recons_video.shape[0],1,1))
mask_binary = np.where(mask > 0.8, 1, 0)

noise = gt_video - recons_video
norm_noise = (noise - noise.min())*255/(noise.max() - noise.min())
masked_noise = noise*mask_binary
masked_norm_noise = norm_noise*mask + np.ones_like(norm_noise)*(1-mask)*255/2


# Noise distribution

plt.hist(masked_noise[masked_noise != 0].flatten(), bins=50, density=False, alpha=0.7, color='blue')
plt.xlabel('Noise Value')
plt.ylabel('Probability Density')
plt.title('Histogram of Noise Values')
# plt.vlines(x=125, ymin=0, ymax=0.03, colors='black', linestyles='--')
plt.savefig(save_path + '/histogram.png')

reconstruct_video(masked_norm_noise, save_path, 'noise_og')

struct_noise = np.where(masked_noise < -50, 150, 0)

reconstruct_video(struct_noise, save_path, 'noise_struct')


# Frequency study

noise_flat = noise.flatten()

fs = 30  # Example: 30 Hz (e.g., if each frame corresponds to a sample in time)

# Compute the PSD using Welch's method.
# nperseg sets the length of each segment; adjust it based on your data length; shorter nperseg -> no low freq information
frequencies, psd = welch(noise_flat, fs=fs, nperseg=1024)

# Plot the PSD on a semilog-y scale for better visualization.
plt.semilogy(frequencies, psd)
plt.xlabel('Frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.title('Power Spectral Density (Welch)')
plt.show()

shutil.copy(current_dir / Path('config.py'), save_path + '/config.py')
