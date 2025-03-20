import numpy as np

class PupilCenterPipeline():
    def __init__(self):
        pass

    def mean_pupil_pos(self, pupil_center):
        x_mean = np.mean(pupil_center[0, :])
        y_mean = np.mean(pupil_center[1, :])
        pupil_center[0, :] = x_mean
        pupil_center[1, :] = y_mean
        return pupil_center

    def invert_pupil_pos(self, pupil_center):
        x_mean = np.mean(pupil_center[0, :])
        y_mean = np.mean(pupil_center[1, :])
        pupil_center[0, :] = - (pupil_center[0, :] - x_mean) + x_mean
        pupil_center[1, :] = - (pupil_center[1, :] - y_mean) + y_mean
        return pupil_center



    def __call__(self, pupil_center, pc_params):
        
        if pc_params['pupil_pos'] == 'original':
            pupil_center = pupil_center

        elif pc_params['pupil_pos'] == 'mean':
            pupil_center = self.mean_pupil_pos(pupil_center)
        
        elif pc_params['pupil_pos'] == 'inverse':
            pupil_center = self.invert_pupil_pos(pupil_center)
        
        return pupil_center


