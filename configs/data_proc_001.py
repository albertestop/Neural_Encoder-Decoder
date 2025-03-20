import os

from src.utils import get_lr
from src import constants


image_size = (64, 64)
batch_size = 32
base_lr = 3e-4
frame_stack_size = 16
animal = 'ESPM126'
session = '2025-02-26_02_ESPM126'
exp_directory = '/home/adamranson/data/Repository/'

data = dict(
    session = session,
    session_dir = os.path.join(exp_directory, animal, session),
    mouse_run = '000',
    stim_eye = 'Left'   # Right or Left
)

videos_params = dict(
    videos_dir = '/home/adamranson/data/vid_for_decoder/all_movie_clips_bv_sets/001/',
)

response_params = dict(
    has_data = True,
    keep_only_spikes = False,
    responses_renorm = False,
    renorm = 'sens_renorm'   # sens_renorm/abs_mean/mean/

)

behavior_params = dict(
    has_speed = True,
    gen_speed_data = 'zeros', # only used if has_speed = False
    has_pupil_dilation = True,
    gen_pupil_data = 'zeros'   # zeros/mean/brightness_reactive
)

pupil_pos_params = dict(
    has_data = True,
    gen_pupil_pos_data = 'zeros', 
)
