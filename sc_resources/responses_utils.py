import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import json


class ResponsesSC:

    def __init__(self, parent_dir, session):
        self.parent_dir = parent_dir
        self.session = session

    def responses_time_coherence(self, trials_time, responses):
        response_mean = np.mean(responses.data, axis=0)
        response_means, time_values = np.array([]), np.array([])
        for t_0 in trials_time:
            indexes = np.where((responses.time_global > t_0 - 8) & (responses.time_global < t_0 + 8))
            response_means = np.concatenate((response_means, response_mean[indexes]))
            time_values = np.concatenate((time_values, responses.time_global[indexes] - t_0))
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
        rec_rate = len(np.where((responses.time_global >= 10) & (responses.time_global < 11))[0])
        plt.plot(time_tot, responses_tot, label='response_mean')
        plt.axhline(y=response_mean_amp, color='green', linestyle='--', label='mean session response amp')
        plt.axvline(x=0, color='red', linestyle='--', label='trial_start')
        plt.ylabel('Response Amplitude')
        plt.xlabel('Time')
        plt.title('Response time coherence')
        plt.legend()
        plt.savefig(os.path.join(self.parent_dir, 'pre_processing', self.session, 'sanity_check', 'responses', 'time_coherence_response.png'))
        plt.cla()


    def sc_plane_neurons(self, responses):
        mask = responses.data != responses.data[:, 0][:, None]
        first_diff_idx = np.argmax(mask, axis=1)
        first_diff_idx[~mask.any(axis=1)] = -1
        neuron_plane = first_diff_idx % responses.n_planes
        unique_values, counts = np.unique(neuron_plane, return_counts=True)
        unique_values= [str(x) for x in unique_values]
        min_n, max_n = min(counts), max(counts)
        if np.abs(max_n - min_n) > 1:
            print('Planes are not distributed evenly')
            print('Plane, num_cells:')
            print(unique_values, counts)
        else:
            print('Planes detected correctly')
        unique_values = [str(x) for x in unique_values]
        counts = [str(x) for x in counts]
        result = dict(zip(unique_values, counts))
        with open(os.path.join(self.parent_dir, 'pre_processing', self.session, 'sanity_check', 'responses', 'plane_neurons.txt'), 'w') as file:
            json.dump(result, file, indent=4)

    def pxl_neuron_act_corr(parent_dir, session, trials_df, responses, videos):
        """
        Plots the linear correlation of a neuron with a pixel.

        Not implemented.
        """
        correlations = []
        trials = np.random.randint(0, len(trials_df), size=20)
        neurons = np.random.randint(0, responses.num_neurons, size=4)
        for trial in trials:
            trial_id = trials_df['F1_name'].iloc[trial]
            trial_t0 = trials_df['time'].iloc[trial]
            trial_duration = trials_df['duration'].iloc[trial]
            indexes = np.where((responses.time_global >= trial_t0) & (responses.time_global < trial_t0 + 30))
            trial_response = responses.data[:, indexes]
            videos.load_video(trial_id, trial_t0, trial_duration)
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
        plt.savefig(os.path.join(parent_dir, 'data', 'data_processing', 'sanity_checks', session + '/neuron_pixel_corr.png'))
        plt.close()

    def sens_lab_comparison(self, sens_response, lab_response, pre_processing=True):
        if not pre_processing:
            check_data= []
            if sens_response.shape == lab_response.shape: check_data.append('Response shapes are equal with shape = ' + str(sens_response.shape))
            else: check_data.append('Response shapes are: sens_response_shape = ' + str(sens_response.shape) + ', lab_response_shape = ' + str(lab_response.shape))
        plt.figure(4)
        plt.plot(np.arange(sens_response.shape[1]), sens_response[3, :], label='Sens neuron response')
        plt.plot(np.arange(lab_response.shape[1]), lab_response[3, :], label='Lab neuron response')
        plt.xlabel('Frame')
        plt.ylabel('Amplitude')
        plt.title('Sensorium vs lab neuron response')
        plt.legend()
        plt.savefig(os.path.join(self.parent_dir, 'pre_processing', self.session, 'sanity_check', 'responses', 'Response_single_SensvsLab_ex1.png'))

        plt.figure(8)
        plt.plot(np.arange(sens_response.shape[1]), sens_response[-10, :], label='Sens neuron response')
        plt.plot(np.arange(lab_response.shape[1]), lab_response[-10, :], label='Lab neuron response')
        plt.xlabel('Frame')
        plt.ylabel('Amplitude')
        plt.title('Sensorium vs lab neuron response')
        plt.legend()
        plt.savefig(os.path.join(self.parent_dir, 'pre_processing', self.session, 'sanity_check', 'responses', 'Response_single_SensvsLab_ex2.png'))

        plt.figure(5)
        plt.imshow(sens_response, aspect='auto', cmap='viridis', vmin=0, vmax=50, interpolation='nearest')
        plt.colorbar(label='Response Intensity')
        plt.xlabel('Frame')
        plt.ylabel('Neuron')
        plt.title('Sens response')
        plt.savefig(os.path.join(self.parent_dir, 'pre_processing', self.session, 'sanity_check', 'responses', 'Response_sens.png'))

        plt.figure(6)
        plt.imshow(lab_response, aspect='auto', cmap='viridis', vmin=0, vmax=50, interpolation='nearest')
        plt.colorbar(label='Response Intensity')
        plt.xlabel('Frame')
        plt.ylabel('Neuron')
        plt.title('Lab response')
        plt.savefig(os.path.join(self.parent_dir, 'pre_processing', self.session, 'sanity_check', 'responses', 'Response_lab.png'))