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

def read_video_gray(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale explicitly (just in case video has 3 channels)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()
    return np.stack(frames, axis=0)  # shape = (T, H, W), dtype=uint8


def _to_THW_uint8_gray(video):
    """Accepta (T,H,W) o (T,1,H,W) en uint8 o float.
       Retorna arr (T,H,W) uint8 i metadata per revertir."""
    meta = {"shape": video.shape, "dtype": video.dtype, "had_channel": False}
    x = video
    # Porta a (T,H,W)
    if x.ndim == 4 and x.shape[1] == 1:
        x = x[:, 0, :, :]
        meta["had_channel"] = True
    assert x.ndim == 3, "Esperava (T,H,W) o (T,1,H,W) per B/N"

    # A uint8 [0..255]
    if x.dtype != np.uint8:
        mx = float(x.max()) if x.size else 1.0
        if np.issubdtype(x.dtype, np.floating):
            x = (x * (255.0 if mx <= 1.0 else 1.0)).clip(0,255).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    return x, meta

def _from_THW_uint8_gray(arr, meta):
    """Reverteix a dtype i shape originals."""
    out = arr
    # Torna a dtype original
    if meta["dtype"] != np.uint8:
        if np.issubdtype(meta["dtype"], np.floating):
            out = (out.astype(np.float32) / 255.0).astype(meta["dtype"])
        else:
            out = out.astype(meta["dtype"])
    # Torna a (T,1,H,W) si cal
    if meta["had_channel"]:
        out = out[:, None, :, :]
    return out

# ------- 1) Denoise espacial (ràpid) -------
def denoise_gaussian_gray(video, ksize=5, sigma=0):
    arr, meta = _to_THW_uint8_gray(video)
    out = np.empty_like(arr)
    for t in range(arr.shape[0]):
        out[t] = cv2.GaussianBlur(arr[t], (ksize, ksize), sigma)
    return _from_THW_uint8_gray(out, meta)

def denoise_median_gray(video, ksize=3):
    arr, meta = _to_THW_uint8_gray(video)
    out = np.empty_like(arr)
    for t in range(arr.shape[0]):
        out[t] = cv2.medianBlur(arr[t], ksize)
    return _from_THW_uint8_gray(out, meta)

# ------- 2) Denoise espacio-temporal (millor qualitat) -------
def denoise_nlm_temporal_gray(video, window=5, h=7, template=7, search=21):
    """OpenCV Non-Local Means multiframe per B/N.
       window: imparell (3,5,7...). h: força de denoise."""
    assert window % 2 == 1 and window >= 3
    arr, meta = _to_THW_uint8_gray(video)
    T = arr.shape[0]
    half = window // 2
    out = np.empty_like(arr)

    for i in range(T):
        idxs = np.clip(np.arange(i - half, i + half + 1), 0, T - 1)
        frames = [arr[k] for k in idxs.tolist()]  # llista de 2D uint8
        den = cv2.fastNlMeansDenoisingMulti(frames, half, window, None, h, template, search)
        out[i] = den
    return _from_THW_uint8_gray(out, meta)


#video = read_video_gray(video_path)
#video = np.load(video_path)[0, 0]
video = tifffile.imread(video_path)
current_dir = str(current_dir)

# 1) Ràpid (soroll lleu / gra suau):
video_suau = denoise_gaussian_gray(video, ksize=3, sigma=0)
video_suau_ups = video_suau.repeat(10, axis=1).repeat(10, axis=2)
iio.imwrite(
    current_dir + '/video_suau.mp4',
    video_suau_ups.astype(np.uint8),
    fps=30,
    codec="libx264",
    ffmpeg_params=["-pix_fmt", "yuv420p"]
)
video_suau2 = denoise_median_gray(video, ksize=3)
video_suau2_ups = video_suau2.repeat(10, axis=1).repeat(10, axis=2)
iio.imwrite(
    current_dir + '/video_suau2.mp4',
    video_suau2_ups.astype(np.uint8),
    fps=30,
    codec="libx264",
    ffmpeg_params=["-pix_fmt", "yuv420p"]
)

# 2) Qualitat millor (soroll moderat/alt, tremolor lleu):
video_netej = denoise_nlm_temporal_gray(video, window=5, h=7, template=7, search=21)
video_netej_ups = video_netej.repeat(10, axis=1).repeat(10, axis=2)
iio.imwrite(
    current_dir + '/video_netej.mp4',
    video_netej_ups.astype(np.uint8),
    fps=30,
    codec="libx264",
    ffmpeg_params=["-pix_fmt", "yuv420p"]
)

video_suau_suau2 = denoise_median_gray(video_suau, ksize=3)
video_suau_suau2_ups = video_suau_suau2.repeat(10, axis=1).repeat(10, axis=2)
iio.imwrite(
    current_dir + '/video_suau_suau2_ups.mp4',
    video_suau_suau2_ups.astype(np.uint8),
    fps=30,
    codec="libx264",
    ffmpeg_params=["-pix_fmt", "yuv420p"]
)
