from pathlib import Path
import os
import numpy as np
import cv2
import math


def generate_lines(angle: float, contrast: float) -> np.ndarray:
    """
    Generate grayscale image with black and white parallel lines.

    Parameters
    ----------
    angle : float
        Angle of the lines in degrees (0° means vertical stripes).
    contrast : float
        Contrast between 0 and 1.
        - 0 -> flat gray (127)
        - 1 -> full black (0) and white (255)
    shape : tuple
        Output image shape (height, width).

    Returns
    -------
    img : np.ndarray
        Grayscale image array of shape (360, 320) with dtype=np.uint8.
    """
    h, w = (360, 320)
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    theta = np.deg2rad(angle)
    proj = x * np.cos(theta) + y * np.sin(theta)

    stripe_pattern = np.sin(2 * np.pi * proj / 20)  # 20 px period

    img = 127.5 + 127.5 * contrast * stripe_pattern
    return img.astype(np.uint8)


def generate_projections(stim, duration, F1_angle, F1_contrast, F2_angle, F2_contrast):
    img = np.zeros((360, 640))
    img[:, :320] = generate_lines(F1_angle, F1_contrast)
    img[:, 320:] = generate_lines(F2_angle, F2_contrast)
    img = np.repeat(img[np.newaxis, :, :], duration*30, axis=0)
    return img
        

def mean_motion_direction_per_frame(video: np.ndarray, coords: str = "math"):
    """
    Per-frame mean motion direction for a grayscale video.

    Parameters
    ----------
    video : np.ndarray
        Shape (T, H, W) for grayscale
    coords : {"math","image"}
        "math": 0°=right, 90°=up   (y axis up)
        "image": 0°=right, 90°=down (y axis down, as in images)

    Returns
    -------
    angles_deg : np.ndarray
        Shape (T,) angles in degrees in [0,360). angles_deg[0] = np.nan (no flow before first frame).
    mean_vecs : np.ndarray
        Shape (T, 2) of (vx, vy) per frame (pixels/frame). First row = (np.nan, np.nan).
    speeds : np.ndarray
        Shape (T,) mean vector magnitudes (pixels/frame). speeds[0] = np.nan.
    """
    vid = video.astype(np.float32)
    if vid.max() > 1.0:
            vid /= 255.0

    T, H, W = vid.shape
    angles_deg = np.full(T, np.nan, dtype=np.float32)
    speeds     = np.full(T, np.nan, dtype=np.float32)
    mean_vecs  = np.full((T, 2), np.nan, dtype=np.float32)

    for t in range(T - 1):
        prev = vid[t]
        next_ = vid[t + 1]
        flow = cv2.calcOpticalFlowFarneback(
            prev, next_, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        vx = flow[..., 0]             # +x right
        vy = flow[..., 1]             # +y down (image coords)
        mean_vx = vx.mean()
        mean_vy = vy.mean()

        if coords == "math":
            # flip y to make +y up
            ang = np.degrees(np.arctan2(-mean_vy, mean_vx)) % 360.0
        elif coords == "image":
            ang = np.degrees(np.arctan2(mean_vy, mean_vx)) % 360.0

        spd = np.hypot(mean_vx, mean_vy)

        angles_deg[t + 1] = ang
        speeds[t + 1] = spd
        mean_vecs[t + 1, 0] = mean_vx
        mean_vecs[t + 1, 1] = mean_vy
    speeds = np.nan_to_num(speeds, nan=0.0)
    speeds = speeds / np.max(speeds)
    mean_vecs = np.nan_to_num(mean_vecs, nan=0.0)
    angles_deg = np.nan_to_num(angles_deg, nan=0.0)

    return angles_deg, mean_vecs, speeds


def arrow_image(x: float,
                y: float,
                units: str, # coords or degs
                h: int = 360,
                w: int = 640,
                thickness: int = 2) -> np.ndarray:
    """
    Genera una imatge (H x W) en blanc i negre amb una fletxa que surt del centre.
    
    Paràmetres
    ----------
    angle_degrees : float
        Angle en graus. 0° apunta cap a la dreta, antihorari.
    module : float
        Longitud de la fletxa en píxels.
    h, w : int
        Alçada i amplada de la imatge (per defecte 360 x 640).
    thickness : int
        Gruix del traç.

    Retorna
    -------
    np.ndarray
        Imatge en escala de grisos (uint8) de mida (h, w).
    """
    # Crear fons negre (1 canal)
    img = np.zeros((h, w), dtype=np.uint8)

    # Centre
    cx, cy = w // 2, h // 2

    # Converteix angle a radians
    if units == 'deg':
        module = module * 160
        theta = math.radians(x)

        dx = y * math.cos(theta)
        dy = y * math.sin(theta)
    else:
        dx, dy = x, y

    x2 = int(round(cx + dx))
    y2 = int(round(cy - dy))

    # Dibuixa la fletxa en blanc (255)
    cv2.arrowedLine(img, (cx, cy), (x2, y2), 255, thickness, tipLength=0.15)

    return img


def mean_mov(arr: np.ndarray, window: int = 15) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    cumsum = np.cumsum(np.insert(arr, 0, 0))  
    
    result = np.empty_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        window_sum = cumsum[i + 1] - cumsum[start]
        result[i] = window_sum  / window
    return result