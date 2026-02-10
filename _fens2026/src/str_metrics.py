import numpy as np
from scipy.stats import linregress
from skimage.metrics import structural_similarity as ssim
import gzip

def temporal_corr(frames: np.ndarray, mask: np.ndarray) -> np.float32:
    """
    Compute the pearson correlation
    Inputs:
        - (frames, x, y)
        - mask (frames, x, y)
    """

    return np.mean([np.corrcoef(f0.ravel()[row_mask0.ravel()], f1.ravel()[row_mask1.ravel()])[0,1]
                    for f0, f1, row_mask0, row_mask1 in zip(frames[:-1], frames[1:], mask[:-1], mask[1:])])

def temporal_ssim(frames, mask):
    return np.mean([ssim(f0.ravel()[row_mask0.ravel()], f1.ravel()[row_mask1.ravel()], data_range=255)
                    for f0, f1, row_mask0, row_mask1 in zip(frames[:-1], frames[1:], mask[:-1], mask[1:])])

def spectral_slope(frames, mask):
    slopes = []
    for frame in frames:
        frame = frame * mask
        F = np.fft.fftshift(np.fft.fft2(frame))
        P = np.abs(F)**2
        y, x = np.indices(P.shape)
        r = np.hypot(x - x.mean(), y - y.mean()).astype(int)
        radial_prof = np.bincount(r.ravel(), P.ravel()) / np.bincount(r.ravel())
        f = np.arange(1, len(radial_prof))
        slope, *_ = linregress(np.log(f), np.log(radial_prof[1:]))
        slopes.append(slope)
    return np.mean(slopes)


def compression_gain(video, mask):
    video = video.ravel()[mask.ravel()]
    pre_weight = video.nbytes
    post_weight = len(gzip.compress(video))
    return pre_weight/ post_weight



if __name__ == '__main__':
    
    print('\nMETRIC STUDY')

    video_path = '/home/albertestop/data/processed_data/sensorium_all_2023/dynamic29515/data/videos/0.npy'
    video = np.load(video_path)
    n_nans = int(np.isnan(video).sum() / (video.shape[0] * video.shape[1]))
    if n_nans > 0: video = video[:, :, :- n_nans]
    video = np.transpose(video, (2, 0, 1))
    mask_path = '/home/albertestop/Sensorium/Clopath/reconstructions/masks/mask_m0.npy'
    mask = np.load(mask_path)
    mask = mask > 0.7
    mask = np.repeat(mask[None, 14:-14, :], video.shape[0], axis=0)
    print('\nFor an actual video:')
    print(video.shape, mask.shape)
    print(f'Temporal correlation: {temporal_corr(video, mask)}')
    print(f'Temporal ssim: {temporal_ssim(video, mask)}')
    print(f'Spectral slope: {spectral_slope(video, mask[0])}')
    print(f'Compression rate: {compression_gain(video, mask)}')

    video_path = '/home/albertestop/Sensorium/Clopath/reconstructions/results/71/dynamic29515_downs_7hz/87/reconstruction_array.npy'
    video = np.load(video_path)
    video = np.transpose(video[0, 0, :, 14:-14, :], axes=(1, 2, 0))
    n_nans = int(np.isnan(video).sum() / (video.shape[0] * video.shape[1]))
    if n_nans > 0: video = video[:, :, :- n_nans]
    video = np.transpose(video, (2, 0, 1))
    print('\nFor a sensorium trial reconstruction:')
    print(video.shape, mask.shape)
    print(f'Temporal correlation: {temporal_corr(video, mask)}')
    print(f'Temporal ssim: {temporal_ssim(video, mask)}')
    print(f'Spectral slope: {spectral_slope(video, mask[0])}')
    print(f'Compression rate: {compression_gain(video, mask)}')

    video_path = '/home/albertestop/Sensorium/Clopath/reconstructions/results/104/2025-04-01_01_ESPM127_002/28/reconstruction_array.npy'
    video = np.load(video_path)
    video = np.transpose(video[0, 0, :, 14:-14, :], axes=(1, 2, 0))
    n_nans = int(np.isnan(video).sum() / (video.shape[0] * video.shape[1]))
    if n_nans > 0: video = video[:, :, :- n_nans]
    video = np.transpose(video, (2, 0, 1))
    mask_path = '/home/albertestop/Sensorium/Clopath/reconstructions/masks/mask_2025-04-01_01_ESPM127_000.npy'
    mask = np.load(mask_path)
    mask = mask > 0.7
    mask = np.repeat(mask[None, 14:-14, :], video.shape[0], axis=0)
    print('\nFor a lab trial reconstruction:')
    print(video.shape, mask.shape)
    print(f'Temporal correlation: {temporal_corr(video, mask)}')
    print(f'Temporal ssim: {temporal_ssim(video, mask)}')
    print(f'Spectral slope: {spectral_slope(video, mask[0])}')
    print(f'Compression rate: {compression_gain(video, mask)}')

    video_path = '/home/albertestop/Sensorium/Clopath/reconstructions/results/25/2025-02-26_02_ESPM126_001/138/reconstruction_array.npy'
    video = np.load(video_path)
    video = np.transpose(video[0, 0, :, 14:-14, :], axes=(1, 2, 0))
    n_nans = int(np.isnan(video).sum() / (video.shape[0] * video.shape[1]))
    if n_nans > 0: video = video[:, :, :- n_nans]
    video = np.transpose(video, (2, 0, 1))
    mask_path = '/home/albertestop/Sensorium/Clopath/reconstructions/masks/mask_m0.npy'
    mask = np.load(mask_path)
    mask = mask > 0.7
    mask = np.repeat(mask[None, 14:-14, :], video.shape[0], axis=0)
    print('\nFor a 0 video corr reconstruction:')
    print(video.shape, mask.shape)
    print(f'Temporal correlation: {temporal_corr(video, mask)}')
    print(f'Temporal ssim: {temporal_ssim(video, mask)}')
    print(f'Spectral slope: {spectral_slope(video, mask[0])}')
    print(f'Compression rate: {compression_gain(video, mask)}')


