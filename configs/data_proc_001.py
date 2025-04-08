import os


image_size = (64, 64)
batch_size = 32
base_lr = 3e-4
frame_stack_size = 16
animal = 'ESPM127'
session = '2025-04-01_01_ESPM127'
exp_directory = '/home/pmateosaparicio/data/Repository/'

data = dict(
    session = session,
    session_dir = os.path.join(exp_directory, animal, session),
    mouse_run = '000',
    eyes = ['Left', 'Right'],
    stim_eye = 'Left',   # Right or Left
    sleep = False
)

videos_params = dict(
    videos_dir = '/home/adamranson/data/vid_for_decoder/all_movie_clips_bv_sets/001',  # Ficar sense '/' al final
    freq = 30
)

response_params = dict(
    has_data = True,
    downscale = True,
    upscale = False,    # Not compatible with downscale
    keep_only_spikes = False, # True not currently working
    resample = True,
    responses_renorm = True,
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
