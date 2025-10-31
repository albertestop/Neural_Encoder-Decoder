import sys
import cv2
from pathlib import Path
import numpy as np
import imageio.v3 as iio
import tifffile

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
reparent_dir = parent_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(str(reparent_dir))

video_path = '/home/albertestop/Sensorium/Clopath/reconstructions/results/155/2025-08-07_01_ESPM163_000/30/optimized_input.tif'


video = tifffile.imread(video_path)
current_dir = str(current_dir)

for i in range(video.shape[0] - 6):
    video[i] = np.mean(video[i:i+7], axis=0)
video = video.repeat(10, axis=1).repeat(10, axis=2)
iio.imwrite(
    current_dir + '/video_mean.mp4',
    video.astype(np.uint8),
    fps=30,
    codec="libx264",
    ffmpeg_params=["-pix_fmt", "yuv420p"]
)
