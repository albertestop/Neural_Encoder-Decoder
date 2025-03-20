import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from src import utils

old_mouse = [
    'dynamic29156', 'dynamic29228', 'dynamic29234', 'dynamic29513', 'dynamic29514'
]

new_mouse = [
    'dynamic29515', 'dynamic29623', 'dynamic29647', 'dynamic29712', 'dynamic29755'
]

sens_mice = old_mouse + new_mouse

sens_data_dir = '/home/albertestop/Sensorium/sensorium/data/sensorium_all_2023'

n_trials = []
n_frames = []
response_n_peaks = []
response_amp_peaks = []
speed = []
stop_time = []
pupil_dilation = []
pupil_center_x = []
pupil_center_y = []
correlation = []


responses = [0]
responses_mean = []

for mouse in sens_mice:
    print('\n' + mouse)
    file_names = [f for f in os.listdir(os.path.join(sens_data_dir, mouse, 'data', 'responses')) if os.path.isfile(os.path.join(sens_data_dir, mouse, 'data', 'responses', f))]
    n_trials.append(len(file_names))
    session_n_frames = []
    session_response_n_peaks = []
    session_response_amp_peaks = []
    session_speed = []
    session_stop_time = []
    session_pupil_dilation = []
    session_pupil_center_x = []
    session_pupil_center_y = []
    session_correlation = []


    count = 0
    for trial in file_names:
        sens_response = np.load(os.path.join(sens_data_dir, mouse, 'data', 'responses', trial))
        if len(responses) == 1: responses = sens_response[:, :200]
        responses = np.vstack((responses, sens_response[:, :200]))

        count += 1
        print(str(count) + '/' + str(len(file_names)) + ', ' + str(trial))
        responses_mean.append(np.mean(responses, axis=0))
    responses = [0]

responses_mean = np.mean(responses_mean, axis=0)
plt.plot(np.arange(0, 8, 1/30), responses_mean[:240])
plt.savefig('plot/data/sens_data_study/responses_mean.png')

"""        sens_video = np.load(os.path.join(sens_data_dir, mouse, 'data', 'videos', trial))
        sens_response = np.load(os.path.join(sens_data_dir, mouse, 'data', 'responses', trial))
        sens_behavior = np.load(os.path.join(sens_data_dir, mouse, 'data', 'behavior', trial))
        sens_pupil_center = np.load(os.path.join(sens_data_dir, mouse, 'data', 'pupil_center', trial))

        # Number of frames
        trial_frames = np.sum(~np.isnan(sens_behavior[0, :]))
        session_n_frames.append(trial_frames)

        # Trial peak amplitude mean and std (within the trial), n_peaks mean and std (between trials), neuron ordering correlation.
        trial_amp_peaks = []
        trial_n_peaks = []
        trial_correlation = []
        for i in range(len(sens_response[:, 0])):
            neuron_response = sens_response[i, :trial_frames]
            neuron_response[neuron_response < 2] = 0
            peak_pos = find_peaks(neuron_response)[0]
            peaks = neuron_response[peak_pos]
            trial_n_peaks.append(len(peaks))
            if len(peaks) > 0: trial_amp_peaks.append([np.mean(peaks), np.std(peaks)])
            if i < len(sens_response[:, 0]) - 1 and len(peaks) > 0:
                trial_correlation.append(np.mean(np.abs(peaks - sens_response[i + 1, peak_pos])))
        session_response_n_peaks.append(np.mean(trial_n_peaks))
        if len(trial_amp_peaks) > 0: session_response_amp_peaks.append([np.mean(np.array(trial_amp_peaks)[:, 0]), np.mean(np.array(trial_amp_peaks)[:, 1])])
        if len(trial_correlation) > 0: session_correlation.append(np.mean(np.array(trial_correlation)))

        # Pupil dilation and speed mean and std (within the trial)
        session_speed.append([np.mean(sens_behavior[1, :trial_frames]), np.std(sens_behavior[1, :trial_frames])])
        trial_stop_time = 0
        for i in range(len(sens_behavior[1, :trial_frames])):
            if sens_behavior[1, i] < max(sens_behavior[1, :trial_frames])/50:
                trial_stop_time += 1
        session_stop_time.append(trial_stop_time)
        session_pupil_dilation.append([np.mean(sens_behavior[0, :trial_frames]), np.std(sens_behavior[0, :trial_frames])])

        # Pupil center
        session_pupil_center_x.append([np.mean(sens_pupil_center[0, :trial_frames]), np.std(sens_pupil_center[0, :trial_frames])])
        session_pupil_center_y.append([np.mean(sens_pupil_center[1, :trial_frames]), np.std(sens_pupil_center[1, :trial_frames])])


    n_frames.append([np.mean(session_n_frames), np.std(session_n_frames)])
    response_amp_peaks.append([np.mean(np.array(session_response_amp_peaks)[:, 0]), np.mean(np.array(session_response_amp_peaks)[:, 1])])
    response_n_peaks.append([np.mean(session_response_n_peaks), np.std(session_response_n_peaks)])
    correlation.append([np.mean(session_correlation), np.std(session_correlation)])
    speed.append([np.mean(np.array(session_speed)[:, 0]), np.mean(np.array(session_speed)[:, 1])])
    stop_time.append([np.mean(session_stop_time), np.std(session_stop_time)])
    pupil_dilation.append([np.mean(np.array(session_pupil_dilation)[:, 0]), np.mean(np.array(session_pupil_dilation)[:, 1])])
    pupil_center_x.append([np.mean(np.array(session_pupil_center_x)[:, 0]), np.mean(np.array(session_pupil_center_x)[:, 1])])
    pupil_center_y.append([np.mean(np.array(session_pupil_center_y)[:, 0]), np.mean(np.array(session_pupil_center_y)[:, 1])])
"""
"""with open('plot/data/sens_data_study/sens_data_study.txt' , "w") as file:
    file.write('SENSORIUM DATA STUDY\n')
    file.write('\nOVERALL MOUSE STATISTICS:')
    file.write('\nMean number of trials: ' + str(np.mean(n_trials)))
    file.write('\nMean number of trial frames: ' + str(np.mean(np.array(n_frames)[:, 0])) + ' ± σ = ' + str(np.mean(np.array(n_frames)[:, 1])) + ' (from trial to trial)')
    file.write('\nResponses:')
    file.write('\nMean amplitude of the responses spikes: ' + str(np.mean(np.array(response_amp_peaks)[:, 0])) + ' ± σ = ' + str(np.mean(np.array(response_amp_peaks)[:, 1])) + ' (within a trial)')
    file.write('\nMean number of spikes in a trial: ' + str(np.mean(np.array(response_n_peaks)[:, 0])) + ' ± σ = ' + str(np.mean(np.array(response_n_peaks)[:, 1])) + ' (from trial to trial)')
    file.write('\nMean peak differences in adjacent neurons: ' + str(np.mean(np.array(correlation)[:, 0])) + ' ± σ = ' + str(np.mean(np.array(correlation)[:, 1])) + ' (from trial to trial)')
    file.write('\nBehavior:')
    file.write('\nMean speed: ' + str(np.mean(np.array(speed)[:, 0])) + ' ± σ = ' + str(np.mean(np.array(speed)[:, 1])) + ' (within a trial)')
    file.write('\nMean trial stop time (frames): ' + str(np.mean(np.array(stop_time)[:, 0])) + ' ± σ = ' + str(np.mean(np.array(stop_time)[:, 1])) + ' (from trial to trial)')
    file.write('\nMean pupil dilation: ' + str(np.mean(np.array(pupil_dilation)[:, 0])) + ' ± σ = ' + str(np.mean(np.array(pupil_dilation)[:, 1])) + ' (within a trial)')
    file.write('\nPupil center:')
    file.write('\nMean x position: ' + str(np.mean(np.array(pupil_center_x)[:, 0])) + ' ± σ = ' + str(np.mean(np.array(pupil_center_x)[:, 1])) + ' (within a trial)')
    file.write('\nMean y position: ' + str(np.mean(np.array(pupil_center_y)[:, 0])) + ' ± σ = ' + str(np.mean(np.array(pupil_center_y)[:, 1])) + ' (within a trial)')

    file.write('\n\nOLD MOUSE STATISTICS:')
    file.write('\nMean number of trials: ' + str(np.mean(n_trials[:len(old_mouse)])))
    file.write('\nMean number of trial frames: ' + str(np.mean(np.array(n_frames)[:len(old_mouse), 0])) + ' ± σ = ' + str(np.mean(np.array(n_frames)[:len(old_mouse), 1])) + ' (from trial to trial)')
    file.write('\nMean peak differences in adjacent neurons: ' + str(np.mean(np.array(correlation)[:len(old_mouse), 0])) + ' ± σ = ' + str(np.mean(np.array(correlation)[:len(old_mouse), 1])) + ' (from trial to trial)')
    file.write('\nResponses:')
    file.write('\nMean amplitude of the responses spikes: ' + str(np.mean(np.array(response_amp_peaks)[:len(old_mouse), 0])) + ' ± σ = ' + str(np.mean(np.array(response_amp_peaks)[:len(old_mouse), 1])) + ' (within a trial)')
    file.write('\nMean number of spikes in a trial: ' + str(np.mean(np.array(response_n_peaks)[:len(old_mouse), 0])) + ' ± σ = ' + str(np.mean(np.array(response_n_peaks)[:len(old_mouse), 1])) + ' (from trial to trial)')
    file.write('\nBehavior:')
    file.write('\nMean speed: ' + str(np.mean(np.array(speed)[:len(old_mouse), 0])) + ' ± σ = ' + str(np.mean(np.array(speed)[:len(old_mouse), 1])) + ' (within a trial)')
    file.write('\nMean trial stop time (frames): ' + str(np.mean(np.array(stop_time)[:len(old_mouse), 0])) + ' ± σ = ' + str(np.mean(np.array(stop_time)[:len(old_mouse), 1])) + ' (from trial to trial)')
    file.write('\nMean pupil dilation: ' + str(np.mean(np.array(pupil_dilation)[:len(old_mouse), 0])) + ' ± σ = ' + str(np.mean(np.array(pupil_dilation)[:len(old_mouse), 1])) + ' (within a trial)')
    file.write('\nPupil center:')
    file.write('\nMean x position: ' + str(np.mean(np.array(pupil_center_x)[:len(old_mouse), 0])) + ' ± σ = ' + str(np.mean(np.array(pupil_center_x)[:len(old_mouse), 1])) + ' (within a trial)')
    file.write('\nMean y position: ' + str(np.mean(np.array(pupil_center_y)[:len(old_mouse), 0])) + ' ± σ = ' + str(np.mean(np.array(pupil_center_y)[:len(old_mouse), 1])) + ' (within a trial)')

    file.write('\n\nNEW MOUSE STATISTICS:')
    file.write('\nMean number of trials: ' + str(np.mean(n_trials[len(old_mouse):])))
    file.write('\nMean number of trial frames: ' + str(np.mean(np.array(n_frames)[len(old_mouse):, 0])) + ' ± σ = ' + str(np.mean(np.array(n_frames)[len(old_mouse):, 1])) + ' (from trial to trial)')
    file.write('\nResponses:')
    file.write('\nMean amplitude of the responses spikes: ' + str(np.mean(np.array(response_amp_peaks)[len(old_mouse):, 0])) + ' ± σ = ' + str(np.mean(np.array(response_amp_peaks)[len(old_mouse):, 1])) + ' (within a trial)')
    file.write('\nMean number of spikes in a trial: ' + str(np.mean(np.array(response_n_peaks)[len(old_mouse):, 0])) + ' ± σ = ' + str(np.mean(np.array(response_n_peaks)[len(old_mouse):, 1])) + ' (from trial to trial)')
    file.write('\nMean peak differences in adjacent neurons: ' + str(np.mean(np.array(correlation)[len(old_mouse):, 0])) + ' ± σ = ' + str(np.mean(np.array(correlation)[len(old_mouse):, 1])) + ' (from trial to trial)')
    file.write('\nBehavior:')
    file.write('\nMean speed: ' + str(np.mean(np.array(speed)[len(old_mouse):, 0])) + ' ± σ = ' + str(np.mean(np.array(speed)[len(old_mouse):, 1])) + ' (within a trial)')
    file.write('\nMean trial stop time (frames): ' + str(np.mean(np.array(stop_time)[len(old_mouse):, 0])) + ' ± σ = ' + str(np.mean(np.array(stop_time)[len(old_mouse):, 1])) + ' (from trial to trial)')
    file.write('\nMean pupil dilation: ' + str(np.mean(np.array(pupil_dilation)[len(old_mouse):, 0])) + ' ± σ = ' + str(np.mean(np.array(pupil_dilation)[len(old_mouse):, 1])) + ' (within a trial)')
    file.write('\nPupil center:')
    file.write('\nMean x position: ' + str(np.mean(np.array(pupil_center_x)[len(old_mouse):, 0])) + ' ± σ = ' + str(np.mean(np.array(pupil_center_x)[len(old_mouse):, 1])) + ' (within a trial)')
    file.write('\nMean y position: ' + str(np.mean(np.array(pupil_center_y)[len(old_mouse):, 0])) + ' ± σ = ' + str(np.mean(np.array(pupil_center_y)[len(old_mouse):, 1])) + ' (within a trial)')

    for i, mouse in enumerate(sens_mice):
        file.write('\n\n' + mouse +  ' MOUSE STATISTICS:')
        file.write('\nMean number of trials: ' + str(n_trials[i]))
        file.write('\nMean number of trial frames: ' + str(np.array(n_frames)[i, 0]) + ' ± σ = ' + str(np.array(n_frames)[i, 1]) + ' (from trial to trial)')
        file.write('\nResponses:')
        file.write('\nMean amplitude of the responses spikes: ' + str(np.array(response_amp_peaks)[i, 0]) + ' ± σ = ' + str(np.array(response_amp_peaks)[i, 1]) + ' (within a trial)')
        file.write('\nMean number of spikes in a trial: ' + str(np.array(response_n_peaks)[i, 0]) + ' ± σ = ' + str(np.array(response_n_peaks)[i, 1]) + ' (from trial to trial)')
        file.write('\nMean peak differences in adjacent neurons: ' + str(np.array(correlation)[i, 0]) + ' ± σ = ' + str(np.array(correlation)[i, 1]) + ' (from trial to trial)')
        file.write('\nBehavior:')
        file.write('\nMean speed: ' + str(np.array(speed)[i, 0]) + ' ± σ = ' + str(np.array(speed)[i, 1]) + ' (within a trial)')
        file.write('\nMean trial stop time (frames): ' + str(np.array(stop_time)[i, 0]) + ' ± σ = ' + str(np.array(stop_time)[i, 1]) + ' (from trial to trial)')
        file.write('\nMean pupil dilation: ' + str(np.array(pupil_dilation)[i, 0]) + ' ± σ = ' + str(np.array(pupil_dilation)[i, 1]) + ' (within a trial)')
        file.write('\nPupil center:')
        file.write('\nMean x position: ' + str(np.array(pupil_center_x)[i, 0]) + ' ± σ = ' + str(np.array(pupil_center_x)[i, 1]) + ' (within a trial)')
        file.write('\nMean y position: ' + str(np.array(pupil_center_y)[i, 0]) + ' ± σ = ' + str(np.array(pupil_center_y)[i, 1]) + ' (within a trial)')
file.close()"""