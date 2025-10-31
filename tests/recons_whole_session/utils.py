import numpy as np
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent


def generate_moving_grating(
    T: int = 360,
    H: int = 640,
    W: int = 360,
    angle: float = 45.0,      # degrees; 0° → vertical stripes, 90° → horizontal stripes
    period: float = 20.0,     # pixels per cycle (stripe pair)
    speed: float = 1.0,       # pixels/frame, motion perpendicular to stripes
    contrast: float = 1.0,    # 0..1 (1 = full black/white, 0 = flat gray)
    phase0: float = 0.0,      # initial phase in radians
    dtype=np.uint8
) -> np.ndarray:
    """
    Generate a video (T,H,W) of a diagonal black–white grating moving
    perpendicular to its stripe direction.

    The image at each (y,x) is:
        I_t(y,x) = 127.5 * [1 + contrast * sin(2π * (x cosθ + y sinθ)/period + φ_t)]
    with φ_t = phase0 + 2π * (t * speed)/period

    Returns
    -------
    video : np.ndarray of shape (T, H, W), dtype=uint8 by default
    """
    # coordinate grid
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # spatial carrier along the normal to the stripes
    theta = np.deg2rad(angle)
    proj = x * np.cos(theta) + y * np.sin(theta)  # (H,W)

    # per-frame phase for motion perpendicular to stripes
    # (advancing phase = shifting pattern along 'proj' direction)
    t = np.arange(T)[:, None, None]               # (T,1,1)
    phases = phase0 + 2.0 * np.pi * (t * speed) / period

    # grating (broadcast over time)
    pattern = np.sin(2.0 * np.pi * proj / period + phases)  # (T,H,W)

    # scale to 0..255 with contrast around mid-gray
    video = 127.5 + 127.5 * np.clip(contrast, 0.0, 1.0) * pattern
    video = np.clip(video, 0, 255)
    print(video.shape)

    return video.astype(dtype)   

if __name__ == "__main__":
    vid = generate_moving_grating(
        T=900, H=720, W=640,
        angle=45,    # diagonal stripes
        period=24,   # pixels per cycle
        speed=1.5,   # pixels per frame perpendicular to stripes
        contrast=1.0
    )
    np.save(current_dir / pathlib.Path('1/video_array.npy'), vid)