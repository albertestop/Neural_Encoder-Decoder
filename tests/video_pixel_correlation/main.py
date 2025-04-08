"""
woked if pasted in ln 153 of Clopath.src.reconstruct.py
"""

import matplotlib.pyplot as plt
import numpy as np
saliency = np.zeros((64, 64))
outputs = self.predictor_withgrads[0](input_prediction)
for i in range(len(outputs)):
    outputs[i, 30].backward(retain_graph=True)
    saliency = saliency + np.array(input_prediction.grad[0, 0, 0].abs().cpu())
    saliency = saliency/np.max(saliency)
plt.imsave('delete.png', saliency, cmap='viridis')
exit()
