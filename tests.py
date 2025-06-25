import numpy as np
import matplotlib.pyplot as plt
from dash import dcc, html, Output, Input, State, ctx
from PIL import Image

FPS: int = 10
DURATION_SEC: int = 10
N_FRAMES: int = FPS * DURATION_SEC

t_video = np.linspace(0, DURATION_SEC, N_FRAMES, endpoint=False)
video = (255 * np.random.rand(N_FRAMES, 180, 320, 3)).astype(np.uint8)
np.save('video.npy', video)