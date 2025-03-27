import numpy as np
import pickle
import os

class PupilPosition:

    def __init__(self, pupil_pos_params, length):
        self.params = pupil_pos_params
        self.session_length = length
    
    def load_data(self, data):
    
        if self.params['has_data']:

            with open(os.path.join(data['session_dir'], 'recordings', 'dlcEye' + data['stim_eye'] + '.pickle'), 'rb') as file:
                file = pickle.load(file)
            pupilX = np.mean(file['pupilX'], axis=1)
            pupilY = np.mean(file['pupilY'], axis=1)
            pupil_pos = np.vstack([pupilX, pupilY])
            
            pupil_time = np.load(os.path.join(data['session_dir'], 'recordings', 'eye_frame_times.npy'))

        else:
            pupil_time = np.arange(0, self.session_length, 1/10)

            if self.params['gen_pupil_pos_data'] == 'zeros':
                pupil_pos = np.zeros((2, len(pupil_time)))

            elif self.params['gen_pupil_pos_data'] == 'sens_mean':
                pupil_pos = np.zeros((2, len(pupil_time)))
                pupil_pos[0, :], pupil_pos[1, :] = 142.963, 139.256



        self.pupilX_mean = np.mean(pupil_pos[0]).astype(np.float32)
        self.pupilY_mean = np.mean(pupil_pos[1]).astype(np.float32)
        self.data = pupil_pos.astype(np.float32)
        self.time = pupil_time.astype(np.float32)


    def process(self, video_frame_time):
        resampled_pupil_pos = []
        for i in range(len(self.data[:, 0])):
            resampled_pupil_pos.append(np.interp(video_frame_time, self.time, self.data[i, :]))
        self.trial_data = np.array(resampled_pupil_pos).astype(np.float32)
        
