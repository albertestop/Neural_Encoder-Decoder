import cv2, numpy as np, skvideo.io
from scipy.stats import linregress
from skimage.metrics import structural_similarity as ssim

def temporal_corr(frames):
    # Pearson correlation on vectorised frames
    return np.mean([np.corrcoef(f0.ravel(), f1.ravel())[0,1]
                    for f0, f1 in zip(frames[:-1], frames[1:])])

def temporal_ssim(frames):
    return np.mean([ssim(f0, f1, data_range=255)
                    for f0, f1 in zip(frames[:-1], frames[1:])])

def spectral_slope(frame):
    F = np.fft.fftshift(np.fft.fft2(frame))
    P = np.abs(F)**2
    y, x = np.indices(P.shape)
    r = np.hypot(x - x.mean(), y - y.mean()).astype(int)
    radial_prof = np.bincount(r.ravel(), P.ravel()) / np.bincount(r.ravel())
    f = np.arange(1, len(radial_prof))
    slope, *_ = linregress(np.log(f), np.log(radial_prof[1:]))
    return slope

def classify(video_path, max_frames=60):
    reader = skvideo.io.vreader(video_path)
    frames = [cv2.cvtColor(next(reader), cv2.COLOR_BGR2GRAY)
              for _ in range(max_frames)]
    tc, ts = temporal_corr(frames), temporal_ssim(frames)
    slope  = np.mean([spectral_slope(f) for f in frames[:10]])
    if (tc < .15 and ts < .10) or abs(slope) < .3:
        return "Pure noise"
    return "Natural video (noisy)"



if __name__ == '__main__':
    video_path = ''
    video = np.load(video_path)