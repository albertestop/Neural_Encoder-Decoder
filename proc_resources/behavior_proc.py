import numpy as np
import pickle
import pandas as pd
import os

from proc_resources.proc_utils import raw_pupil_file


class Behavior:

    def __init__(self, behavior_params, length):
        self.params = behavior_params
        self.session_length = length
        self.mask_threshold = 0.1

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

        self.speed = self.speed.astype(np.float32)
        self.speed_time = self.speed_time.astype(np.float32)
        self.pupil_dilation = self.pupil_dilation.astype(np.float32)
        self.pupil_dilation_time = self.pupil_dilation_time.astype(np.float32)
        self.params = self.params

        self.acceptable_mask = []
        for eye in data['eyes']:
            likelihood_pupil, quality_control = self.load_pupil_raw_likelihood(data, eye)
            acceptable_mask_par = self.mask_creator(likelihood_pupil, quality_control)
            self.acceptable_mask.append(acceptable_mask_par)


    def load_pupil_raw_likelihood(self, data, eye):
        raw_pupil_dir = raw_pupil_file(data['session_dir'], data['session'], eye)
        h5_file_path = os.path.join(data['session_dir'], raw_pupil_dir)
        pupil_data_raw = pd.read_hdf(h5_file_path)
        likelihood_cols = pupil_data_raw.filter(like='likelihood')
        likelihood_pupil = likelihood_cols.filter(like='pupil')
        

        with open(os.path.join(data['session_dir'], 'recordings', 'dlcEye' + data['stim_eye'] + '.pickle'), 'rb') as file:
                file = pickle.load(file)

        quality_control = file['qc']
        return likelihood_pupil, quality_control


    def mask_creator(self, likelihoods, qc):
        avg_likelihood = np.mean(likelihoods, axis=1)
        mask = 1 - avg_likelihood + qc
        acceptable_mask = mask < self.mask_threshold
        return acceptable_mask


    def gen_pupil_dilation(self, trial_video):
        
        frame_bright = trial_video.mean(axis=(0,1))
        center = 20 + (35 - (35 * frame_bright / 255))
        pupil_diameter = np.random.normal(loc=center, scale=0.5, size=len(center))
        self.trial_data[0] = pupil_diameter


    def remake_bad_data(self):
        excluded = np.where(~self.acceptable_mask, self.pupil_dilation, np.nan)
        x = np.arange(len(self.pupil_dilation))
        nans= np.isnan(excluded)
        self.pupil_dilation = np.interp(x[nans], x[~nans], self.pupil_dilation[~nans])  


    def process(self, video_frame_time, trial_video):

        self.remake_bad_data()

        resampled_speed = np.interp(video_frame_time, self.speed_time, self.speed)
        resampled_pupil_dilation = np.interp(video_frame_time, self.pupil_dilation_time, self.pupil_dilation)
        self.trial_data = np.vstack([resampled_pupil_dilation, resampled_speed])

        if not self.params['has_pupil_dilation'] and self.params['gen_pupil_data'] == 'brightness_reactive':
            self.gen_pupil_dilation(trial_video)
        
        self.trial_data = self.trial_data.astype(np.float32)
        