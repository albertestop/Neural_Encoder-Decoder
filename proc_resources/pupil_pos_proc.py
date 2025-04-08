import numpy as np
import pickle
import os
import pandas as pd

from proc_resources.proc_utils import raw_pupil_file

class PupilPosition:

    def __init__(self, pupil_pos_params, length):
        self.params = pupil_pos_params
        self.session_length = length
        self.mask_threshold = 0.1
    
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

        self.acceptable_mask = []
        for eye in data['eyes']:
            likelihood_pupil, quality_control = self.load_pupil_raw_likelihood(data, eye)
            acceptable_mask_par = self.mask_creator(likelihood_pupil, quality_control)
            self.acceptable_mask.append(acceptable_mask_par)

    
    def mask_creator(self, likelihoods, qc):
        avg_likelihood = np.mean(likelihoods, axis=1)
        mask = 1 - avg_likelihood + qc
        acceptable_mask = mask < self.mask_threshold
        return acceptable_mask


    def remake_bad_data(self):
        for i in range(2):
            excluded = np.where(~self.acceptable_mask, self.data[i], np.nan)
            x = np.arange(len(self.data[i]))
            nans= np.isnan(excluded)
            self.data[i, :] = np.interp(x[nans], x[~nans], self.data[i, ~nans])        


    def process(self, video_frame_time):
        self.remake_bad_data()

        resampled_pupil_pos = []
        for i in range(len(self.data[:, 0])):
            resampled_pupil_pos.append(np.interp(video_frame_time, self.time, self.data[i, :]))
        self.trial_data = np.array(resampled_pupil_pos).astype(np.float32)
        

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