import numpy as np
import matplotlib.pyplot as plt

parent_dir = 'Clopath/reconstructions/masks/'
session = 'whole'
video_win = [36, 64]

mask = np.ones((64, 64))


"""center_x = int(video_win[0] / 2) + ((36 - video_win[0]))
center_y = int(video_win[1] / 2) + ((64 - video_win[1]))

mask[center_x - int(video_win[0] / 2) + 14:center_x + int(video_win[0] / 2) + 14, 
    center_y - int(video_win[1] / 2):center_y + int(video_win[1] / 2)] = 1"""

np.save(parent_dir + 'mask_m_' + session + '.npy', mask)
plt.imshow(mask)
plt.colorbar()
plt.savefig(parent_dir + 'mask_summary_m_' + session + '.png')