import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_data_file(plot):
    file_ext = plot[0].split('.')[-1]
    if file_ext == 'npy':
        return np.load(plot[0])
    else:
        raise ValueError('Code not prepared for these file type.')


def load_rasterplot(plot):
    raw_data = load_data_file(plot)
    cfp = Path(__file__).parent.resolve()
    plt.imsave(cfp / 'temp' / "image.png", raw_data)
    img =  plt.imread(cfp / 'temp' / "image.png")
    return img[:, :, :3]


def load_data(plots):
    data = []
    for plot in plots:
        if plot[1] == 'rp':
            par_data = load_rasterplot(plot)
            print(par_data.shape)
        else:
            par_data = load_data_file(plot)
        
        data.append(par_data)

    return data

