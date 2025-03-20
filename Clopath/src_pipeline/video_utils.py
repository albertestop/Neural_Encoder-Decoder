import numpy as np

class VideoPipeline():
    def __init__(self):
        pass

    def supress_pixels(self, video):
        height, width = video.shape[0], video.shape[1]
        rand_height = int(np.random.uniform(0, height))
        rand_width = int(np.random.uniform(0, width))
        video[rand_height, rand_width, :] = 0
        return video

    def __call__(self, video, video_params):
        
        if video_params['pixel_supression'] > 0: 
            video = self.supress_pixels(video)
        
        return video


