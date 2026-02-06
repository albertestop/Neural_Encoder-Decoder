import numpy as np
from PIL import Image


def trial_names(trials_df):
    """
    If the F1_name in the experiment_all_trials.csv does not end in a number,
    we assume that the experiment uses the same video in all the experiments,
    defined in config.video_params.videos_dir()
    """

    if trials_df['F1_name'].iloc[1][-1].isdigit():
        trials_df['F1_name'] = trials_df['F1_name'].str[-5:] + '/'
    else: 
        trials_df['F1_name'] = ''
    
    return trials_df['F1_name']


def load_video(video_id, duration, videos_dir):

    video_lab = []
    for i in range(duration * 30):
        frame_id = str((i // 100)%10) + str((i // 10)%10) + str(i%10)
        frame = load_frame(video_id, frame_id, videos_dir)
        frame = frame.resize((64, 36))
        video_lab.append(np.array(frame))

    trial_video = np.transpose(np.array(video_lab).astype(np.float32), (1, 2, 0)).astype(np.float32)
    return trial_video


def load_frame(video_id, frame_id, videos_dir):
    frame_path = video_id + 'frame-' + frame_id + '.jpg'
    frame = Image.open(videos_dir + '/' + frame_path)
    return frame
