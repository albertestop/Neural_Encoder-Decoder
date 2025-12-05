import numpy as np
from scipy.ndimage import binary_erosion


mask = np.load('/home/albertestop/Sensorium/Clopath/reconstructions/masks/mask_2025-07-04_04_ESPM154_004.npy')
mask = mask[14:-14, :]
mask[mask > 0.75] = 1
mask[mask < 0.75] = 0
mask = mask.repeat(10, axis=0).repeat(10, axis=1)
mask = mask.astype(bool)
print(mask.shape)

edge = mask & ~binary_erosion(mask, structure=np.ones((3,3), dtype=bool))

ys, xs = np.nonzero(edge)
coords_xy = np.column_stack([xs, ys])

np.save('/home/albertestop/Sensorium/tests/mask_edge_coords/mask_edges.npy', coords_xy)
print(coords_xy.shape)