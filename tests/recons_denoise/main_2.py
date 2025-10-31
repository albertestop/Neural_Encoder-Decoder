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

def _to_THW_uint8_gray(video):
    meta = {"shape": video.shape, "dtype": video.dtype, "had_channel": False}
    x = video
    if x.ndim == 4 and x.shape[1] == 1:  # (T,1,H,W) -> (T,H,W)
        x = x[:, 0]
        meta["had_channel"] = True
    assert x.ndim == 3, "Expected (T,H,W) or (T,1,H,W)"
    if x.dtype != np.uint8:
        mx = float(x.max()) if x.size else 1.0
        if np.issubdtype(x.dtype, np.floating):
            x = (x * (255.0 if mx <= 1.0 else 1.0)).clip(0,255).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    return x, meta

def _from_THW_uint8_gray(arr, meta):
    out = arr
    if meta["dtype"] != np.uint8:
        if np.issubdtype(meta["dtype"], np.floating):
            out = (out.astype(np.float32) / 255.0).astype(meta["dtype"])
        else:
            out = out.astype(meta["dtype"])
    if meta["had_channel"]:
        out = out[:, None]
    return out

def warp_to_ref(frame, ref):
    """Warp 'frame' onto 'ref' using dense optical flow (Farnebäck)."""
    flow = cv2.calcOpticalFlowFarneback(frame, ref,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    H, W = ref.shape
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (grid_x + flow[...,0]).astype(np.float32)
    map_y = (grid_y + flow[...,1]).astype(np.float32)
    warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped

def denoise_temporal_motion_comp_gray(video, window=5, spatial_polish=True,
                                      temporal_method="median", mean_sigma=1.5):
    """
    Motion-compensated temporal denoise for grayscale video.
    - window: odd number of frames used around each center frame (e.g., 5 or 7)
    - temporal_method: "median" (robust) or "mean" (with optional pre-blur)
    - spatial_polish: optional small bilateral to clean residual high-freq noise
    """
    assert window % 2 == 1 and window >= 3
    arr, meta = _to_THW_uint8_gray(video)
    T, H, W = arr.shape
    half = window // 2
    out = np.empty_like(arr)

    for t in range(T):
        ref = arr[t]
        idxs = np.clip(np.arange(t - half, t + half + 1), 0, T - 1)
        aligned = []
        for k in idxs:
            fr = arr[k]
            if k == t:
                aligned.append(ref)
            else:
                aligned.append(warp_to_ref(fr, ref))
        stack = np.stack(aligned, axis=0)  # (window, H, W)

        if temporal_method == "median":
            den = np.median(stack, axis=0).astype(np.uint8)
        elif temporal_method == "mean":
            # Light blur to suppress sharp HF noise before averaging
            if mean_sigma and mean_sigma > 0:
                stack_blur = np.stack([cv2.GaussianBlur(f, (0,0), mean_sigma) for f in stack], axis=0)
            else:
                stack_blur = stack
            den = np.mean(stack_blur, axis=0).astype(np.uint8)
        else:
            raise ValueError("temporal_method must be 'median' or 'mean'")

        if spatial_polish:
            # very mild bilateral (keeps edges) – tune as needed
            den = cv2.bilateralFilter(den, d=5, sigmaColor=30, sigmaSpace=30)

        out[t] = den

    return _from_THW_uint8_gray(out, meta)


video = tifffile.imread(video_path)
current_dir = str(current_dir)
# video: np.ndarray (T,H,W) or (T,1,H,W)
clean = denoise_temporal_motion_comp_gray(
    video,
    window=3,                 # try 5 or 7
    temporal_method="mean", # median handles HF flicker very well
    spatial_polish=True       # small edge-preserving polish
)

clean_ups = clean.repeat(10, axis=1).repeat(10, axis=2)
iio.imwrite(
    current_dir + '/clean.mp4',
    clean_ups.astype(np.uint8),
    fps=30,
    codec="libx264",
    ffmpeg_params=["-pix_fmt", "yuv420p"]
)