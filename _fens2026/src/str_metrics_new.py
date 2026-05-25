import numpy as np
from scipy import fftpack
from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import zlib


# ============================================================
# Utility functions
# ============================================================


def _safe_corr(x, y):
    """
    Pearson correlation robust to degenerate cases.
    """
    if len(x) < 2:
        return np.nan

    sx = np.std(x)
    sy = np.std(y)

    if sx < 1e-12 or sy < 1e-12:
        return np.nan

    return np.corrcoef(x, y)[0, 1]


# ============================================================
# 1. Temporal correlation
# ============================================================

def temporal_correlation(frames, mask):
    """
    Mean correlation between consecutive frames.

    Returns:
        mean_corr : float
    """

    corrs = []

    for t in range(len(frames) - 1):
        x, y = frames[t][mask[t]], frames[t + 1][mask[t + 1]]

        corr = _safe_corr(x, y)

        if not np.isnan(corr):
            corrs.append(corr)

    return np.mean(corrs) if len(corrs) > 0 else np.nan


# ============================================================
# 2. Temporal autocorrelation
# ============================================================

def temporal_autocorrelation(frames, mask, lag=1):
    """
    Computes autocorrelation at a given temporal lag.
    """

    corrs = []

    for t in range(len(frames) - lag):
        x, y = frames[t][mask[t]], frames[t + lag][mask[t + lag]]

        corr = _safe_corr(x, y)

        if not np.isnan(corr):
            corrs.append(corr)

    return np.mean(corrs) if len(corrs) > 0 else np.nan


# ============================================================
# 3. Spatial spectral slope
# ============================================================

def spectral_slope_new(frames, mask):
    """
    Estimate average power spectrum slope.

    Natural images/videos:
        slope < 0

    White noise:
        slope ~ 0
    """

    slopes = []

    for frame, m in zip(frames, mask):

        f = np.zeros_like(frame)
        f[m] = frame[m]

        F = np.abs(fftpack.fftshift(fftpack.fft2(f))) ** 2

        h, w = F.shape
        cy, cx = h // 2, w // 2

        y, x = np.indices((h, w))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        r = r.astype(np.int32)

        radial_mean = np.bincount(
            r.ravel(),
            weights=F.ravel()
        ) / np.maximum(np.bincount(r.ravel()), 1)

        freqs = np.arange(1, len(radial_mean))
        power = radial_mean[1:]

        valid = power > 0

        if np.sum(valid) < 10:
            continue

        logf = np.log(freqs[valid]).reshape(-1, 1)
        logp = np.log(power[valid])

        reg = LinearRegression().fit(logf, logp)

        slopes.append(reg.coef_[0])

    return np.mean(slopes) if len(slopes) > 0 else np.nan


# ============================================================
# 4. Entropy
# ============================================================

def frame_entropy(frames, mask, bins=256):
    """
    Mean Shannon entropy over frames.

    Higher entropy => more noise-like.
    """

    entropies = []

    for frame, m in zip(frames, mask):

        vals = frame[m]

        if len(vals) == 0:
            continue

        hist, _ = np.histogram(vals, bins=bins, density=True)

        hist = hist[hist > 0]

        entropies.append(entropy(hist))

    return np.mean(entropies) if len(entropies) > 0 else np.nan


# ============================================================
# 5. Compression ratio
# ============================================================

def compression_ratio(frames, mask):
    """
    Compression ratio using zlib.

    Noise compresses poorly.
    Real videos compress better.
    """

    data = []

    for frame, m in zip(frames, mask):

        tmp = np.zeros_like(frame)
        tmp[m] = frame[m]

        data.append(tmp)

    arr = np.stack(data)

    raw = arr.tobytes()
    compressed = zlib.compress(raw)

    return len(raw) / len(compressed)


# ============================================================
# 6. PCA singular value concentration
# ============================================================

def pca_energy_ratio(frames, mask, n_components=5):
    """
    Measures low-dimensional structure.

    Real videos:
        high concentration in first PCs

    Noise:
        flatter spectrum
    """

    X = []

    for frame, m in zip(frames, mask):

        tmp = np.zeros_like(frame)
        tmp[m] = frame[m]

        X.append(tmp.flatten())

    X = np.stack(X)

    pca = PCA(n_components=min(n_components, len(frames)))
    pca.fit(X)

    return np.sum(pca.explained_variance_ratio_)


# ============================================================
# 7. Frame predictability
# ============================================================

def frame_predictability(frames, mask):
    """
    Linear predictability of next frame from current frame.

    Lower MSE => more structure.
    """

    mses = []

    for t in range(len(frames) - 1):

        x, y = frames[t][mask[t]], frames[t + 1][mask[t + 1]]

        if len(x) < 10:
            continue

        x = x.reshape(-1, 1)

        reg = LinearRegression()
        reg.fit(x, y)

        pred = reg.predict(x)

        mse = mean_squared_error(y, pred)

        mses.append(mse)

    return np.mean(mses) if len(mses) > 0 else np.nan


# ============================================================
# 8. Temporal difference energy
# ============================================================

def temporal_difference_energy(frames, mask):
    """
    Average frame-to-frame difference energy.

    Noise:
        large differences

    Structured videos:
        smaller differences
    """

    diffs = []

    for t in range(len(frames) - 1):

        valid = mask[t] & mask[t + 1]

        if np.sum(valid) == 0:
            continue

        d = frames[t + 1][valid] - frames[t][valid]

        diffs.append(np.mean(d ** 2))

    return np.mean(diffs) if len(diffs) > 0 else np.nan