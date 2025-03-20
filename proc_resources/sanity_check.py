import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from configs.data_proc_001 import data

def sanity_check(parent_dir, trials_df, responses, behavior, pupil_pos, videos):

    os.makedirs(os.path.join(parent_dir, 'data', 'data_processing', 'sanity_checks', data['session']), exist_ok=True)

    print('\nRunning Sanity Check')

    # Time coherence
    t_i, t_f = np.array(trials_df['time'])[0], np.array(trials_df['time'])[-1]
    if t_i - 8 > responses.time[0] and t_i - 8 > behavior.speed_time[0] and t_i - 8 > behavior.pupil_dilation_time[0] and t_i - 8 > pupil_pos.time[0]:
        t_i_check = True 
    else: 
        t_i_check = False
    if t_f + 8 < responses.time[-1] and t_f + 8 < behavior.speed_time[-1] and t_f + 8 < behavior.pupil_dilation_time[-1] and t_f + 8 < pupil_pos.time[-1]:
        t_f_check = True
    else: t_f_check = False
    print('Recording start time and end time checks: ' + str(t_i_check) + ', ' + str(t_f_check))
    response_synchro = (len(responses.data[0, :]) == len(responses.time))
    speed_synchro = (len(behavior.speed) == len(behavior.speed_time))
    pupil_dilation_synchro = (len(behavior.pupil_dilation) == len(behavior.pupil_dilation_time))
    pupil_pos_synchro = (len(pupil_pos.data[0, :]) == len(pupil_pos.time))

    print('Data synchonized: ' + str((response_synchro and speed_synchro and pupil_dilation_synchro and pupil_pos_synchro)))
    if not response_synchro: print('Response data not synchronyzed. Number of time inputs = ' + str(len(responses.time)) + ', number of response inputs = ' + str(len(responses.data[0, :])))
    if not speed_synchro: print('Speed data not synchronyzed. Number of time inputs = ' + str(len(behavior.speed_time)) + ', number of speed inputs = ' + str(len(behavior.speed)))
    if not pupil_dilation_synchro: print('Pupil dilation data not synchronyzed. Number of time inputs = ' + str(len(behavior.pupil_dilation_time)) + ', number of pupil dilation inputs = ' + str(len(behavior.pupil_dilation)))
    if not pupil_pos_synchro: print('Pupil position data not synchronyzed. Number of time inputs = ' + str(len(pupil_pos.time)) + ', number of pupil position inputs = ' + str(len(pupil_pos.data[0, :])))


    # RESPONSES
    # Time Coherence
    response_mean = np.mean(responses.data, axis=0)
    response_means, time_values = np.array([]), np.array([])
    for t_0 in trials_df['time'].iloc[0:]:
        indexes = np.where((responses.time > t_0 - 8) & (responses.time < t_0 + 8))
        response_means = np.concatenate((response_means, response_mean[indexes]))
        time_values = np.concatenate((time_values, responses.time[indexes] - t_0))
    df = pd.DataFrame({
        'responses': response_means,
        'time': time_values
    })
    df.sort_values(by='time', ascending=True, inplace=True)
    df['time'] = (df['time'] / 0.05).round() * 0.05
    df = df.groupby('time', as_index=False)['responses'].mean()
    responses_tot = np.array(df['responses'])
    time_tot = np.array(df['time'])
    response_mean_amp = np.mean(responses_tot)
    rec_rate = len(np.where((responses.time >= 10) & (responses.time < 11))[0])
    plt.plot(time_tot, responses_tot, label='response_mean')
    plt.axhline(y=response_mean_amp, color='green', linestyle='--', label='mean session response amp')
    plt.axvline(x=0, color='red', linestyle='--', label='trial_start')
    plt.ylabel('Response Amplitude')
    plt.xlabel('Time')
    plt.title('Response time coherence')
    plt.legend()
    plt.savefig(os.path.join(parent_dir, 'data', 'data_processing', 'sanity_checks', data['session'] + '/time_coherence_response.png'))
    plt.cla()
    # Pixel-neuron activation corr
    correlations = []
    trials = np.random.randint(0, len(trials_df), size=20)
    neurons = np.random.randint(0, responses.num_neurons, size=4)
    for trial in trials:
        trial_id = trials_df['F1_name'].iloc[trial]
        trial_t0 = trials_df['time'].iloc[trial]
        indexes = np.where((responses.time >= trial_t0) & (responses.time < trial_t0 + 30))
        trial_response = responses.data[:, indexes]
        videos.load_video(trial_id, trial_t0)
        for neuron in neurons:
            for j in range(len(videos.trial_video[:, 0, 0])):
                for k in range(len(videos.trial_video[0, :, 0])):
                    correlations.append(np.array(np.corrcoef(trial_response[neuron, :], videos.trial_video[j, k, :])[0, 1]))
    correlations = np.array(correlations)
    correlations = correlations.reshape(len(trials), len(neurons), len(videos.trial_video[:, 0, 0]), len(videos.trial_video[0, :, 0]))
    correlations_mean = np.mean(correlations, axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    im0 = axes[0, 0].imshow(correlations_mean[0, :, :], cmap='gray')
    axes[0, 1].imshow(correlations_mean[1, :, :], cmap='gray')
    axes[1, 0].imshow(correlations_mean[2, :, :], cmap='gray')
    axes[1, 1].imshow(correlations_mean[3, :, :], cmap='gray')
    fig.colorbar(im0, ax=axes, location='right', shrink=0.8)
    fig.suptitle('Neuron to pixel correlation', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(parent_dir, 'data', 'data_processing', 'sanity_checks', data['session'] + '/neuron_pixel_corr.png'))
    plt.close()

    # Speed
    speed_means = []
    for t_0 in trials_df['time']:
        indexes = np.where((behavior.speed_time > t_0 - 8) & (behavior.speed_time < t_0 + 8))
        speed_means.append(behavior.speed[indexes])
    min_len = min([len(sublist) for sublist in speed_means])
    speed_means = [sublist[:min_len] for sublist in speed_means]
    speed_tot = np.mean(speed_means, axis=0)
    speeed_mean = np.mean(speed_tot)
    plt.plot(np.arange(-8, 8, 16/len(speed_tot)), speed_tot, label='speed_mean')
    plt.axhline(y=speeed_mean, color='green', linestyle='--', label='mean session speed')
    plt.axvline(x=0, color='red', linestyle='--', label='trial_start')
    plt.ylabel('Speed')
    plt.xlabel('Time')
    plt.title('Speed time coherence')
    plt.legend()
    plt.savefig(os.path.join(parent_dir, 'data', 'data_processing', 'sanity_checks', data['session'] + '/time_coherence_speed.png'))
    plt.cla()

    # Pupil dilation
    pupil_means = []
    for t_0 in trials_df['time']:
        indexes = np.where((behavior.pupil_dilation_time > t_0 - 8) & (behavior.pupil_dilation_time < t_0 + 8))
        pupil_means.append(behavior.pupil_dilation[indexes])
    min_len = min([len(sublist) for sublist in pupil_means])
    pupil_means = [sublist[:min_len] for sublist in pupil_means]
    pupil_tot = np.mean(pupil_means, axis=0)
    pupil_mean = np.mean(pupil_tot)
    rec_rate = len(np.where((behavior.pupil_dilation_time >= 10) & (behavior.pupil_dilation_time < 11))[0])
    plt.plot(np.arange(-8, 8, 1/(len(pupil_tot)/16)), pupil_tot, label='pupil_dilation_mean')
    plt.axhline(y=pupil_mean, color='green', linestyle='--', label='mean session pupil dilaion')
    plt.axvline(x=0, color='red', linestyle='--', label='trial_start')
    plt.ylabel('Pupil_dilation')
    plt.xlabel('Time')
    plt.title('Pupil dilation time coherence')
    plt.legend()
    plt.savefig(os.path.join(parent_dir, 'data', 'data_processing', 'sanity_checks', data['session'] + '/time_coherence_pupil_dilation.png'))
    plt.cla()

    # Pupil position
    pupil_pos_means = []
    x_gradient = np.gradient(pupil_pos.data[0, :])
    y_gradient = np.gradient(pupil_pos.data[1, :])
    eye_movement = np.sqrt(np.power(x_gradient, 2) + np.power(y_gradient, 2))
    for t_0 in trials_df['time']:
        indexes = np.where((pupil_pos.time > t_0 - 8) & (pupil_pos.time < t_0 + 8))
        pupil_pos_means.append(eye_movement[indexes])
    min_len = min([len(sublist) for sublist in pupil_pos_means])
    pupil_pos_means = [sublist[:min_len] for sublist in pupil_pos_means]
    eye_movement_tot = np.mean(pupil_pos_means, axis=0)
    x_pos_mean = np.mean(pupil_pos.data[0, :])
    y_pos_mean = np.mean(pupil_pos.data[1, :])
    rec_rate = len(np.where((pupil_pos.time >= 10) & (pupil_pos.time < 11))[0])
    plt.plot(np.arange(-8, 8, 1/(len(pupil_tot)/16)), eye_movement_tot, label='eye_movement_mean')
    plt.axvline(x=0, color='red', linestyle='--', label='trial_start')
    plt.ylabel('Eye movement')
    plt.xlabel('Time')
    plt.title('Eye position time coherence')
    plt.legend()
    plt.savefig(os.path.join(parent_dir, 'data', 'data_processing', 'sanity_checks', data['session'] + '/time_coherence_pupil_pos.png'))
    plt.cla()