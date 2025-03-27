from PIL import Image
import numpy as np
import pandas as pd
import os


class Video:

    def __init__(self, video_params):
        self.params = video_params

    def load_video(self, video_id, trial_t0, duration):
        """
        Loads video from .jpg frames to 3d numpy array with shape = (rows, cols, frame) to match sensorium format. 30s at 30Hz.
        Returns the video array and a time array with the time corresponding to each fotogram.
        """
        video_lab = []
        for i in range(duration * self.params['freq']):
            frame_id = str((i // 100)%10) + str((i // 10)%10) + str(i%10)
            frame = self.load_frame(video_id, frame_id)
            frame = frame.resize((64, 36))
            video_lab.append(np.array(frame))

        self.trial_video = np.transpose(np.array(video_lab).astype(np.float32), (1, 2, 0)).astype(np.float32)
        self.trial_frame_time = np.arange(trial_t0, trial_t0 + duration, 1/self.params['freq']).astype(np.float32)


    def load_frame(self, video_id, frame_id):
        frame_path = video_id + 'frame-' + frame_id + '.jpg'
        frame = Image.open(self.params['videos_dir'] + '/' + frame_path)
        return frame



