import numpy as np
import pickle
import pandas as pd
import os

class Behavior:

    def __init__(self, behavior_params, length):
        self.params = behavior_params
        self.session_length = length

    def load_data(self, data):
        
        if self.params['has_speed']:
            with open(os.path.join(data['session_dir'], 'recordings', 'wheel.pickle'), 'rb') as file:
                file = pickle.load(file)
            self.speed_time = np.array(file['t'])
            self.speed = np.array(file['speed'])
        else:
            if self.params['gen_speed_data'] == 'zeros':
                self.speed_time = np.arange(0, self.session_length, 1/10)
                self.speed = np.zeros(self.speed_time.shape)

        if self.params['has_pupil_dilation']:
            with open(os.path.join(data['session_dir'], 'recordings', 'dlcEye' + data['stim_eye'] + '.pickle'), 'rb') as file:
                file = pickle.load(file)
            pupil_dilation = file['radius']
            pupil_dilation = pd.Series(pupil_dilation).interpolate(method='linear').to_numpy()
            self.pupil_dilation = np.clip(pupil_dilation, None, 100)
            self.pupil_dilation_time = np.load(os.path.join(data['session_dir'], 'recordings/eye_frame_times.npy'))
        else:
            if self.params['gen_pupil_data'] == 'zeros' or self.params['gen_pupil_data'] == 'brightness_reactive':
                self.pupil_dilation = np.zeros(self.speed.shape)
            if self.params['gen_pupil_data'] == 'mean':
                self.pupil_dilation = np.full(self.speed.shape, 30)
                
            self.pupil_dilation_time = np.arange(0, self.session_length, 1/10)

        
        self.params = self.params


    def gen_pupil_dilation(self, trial_video):
        
        frame_bright = trial_video.mean(axis=(0,1))
        center = 20 + (35 - (35 * frame_bright / 255))
        pupil_diameter = np.random.normal(loc=center, scale=0.5, size=len(center))
        self.trial_data[0] = pupil_diameter


    def process(self, video_frame_time, trial_video):

        resampled_speed = np.interp(video_frame_time, self.speed_time, self.speed)
        resampled_pupil_dilation = np.interp(video_frame_time, self.pupil_dilation_time, self.pupil_dilation)
        self.trial_data = np.vstack([resampled_pupil_dilation, resampled_speed])

        if not self.params['has_pupil_dilation'] and self.params['gen_pupil_data'] == 'brightness_reactive':
            self.gen_pupil_dilation(trial_video)