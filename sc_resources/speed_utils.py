import numpy as np
import matplotlib.pyplot as plt
import os


class SpeedSC:

    def __init__(self, parent_dir, session):
        self.parent_dir = parent_dir
        self.session = session

    def speed_time_coherence(self, trials_time, behavior):
        speed_means = []
        for t_0 in trials_time:
            indexes = np.where((behavior.speed_time > t_0 - 8) & (behavior.speed_time < t_0 + 8))
            speed_means.append(behavior.speed[indexes])
        min_len = min([len(sublist) for sublist in speed_means])
        speed_means = [sublist[:min_len] for sublist in speed_means]
        speed_tot = np.mean(np.array(speed_means), axis=0)
        speed_mean = np.mean(speed_tot)
        plt.clf()
        plt.plot(np.arange(-8, 8, 16/len(speed_tot)), speed_tot, label='speed_mean')
        plt.axhline(y=speed_mean, color='green', linestyle='--', label='mean session speed')
        plt.axvline(x=0, color='red', linestyle='--', label='trial_start')
        plt.ylabel('Speed')
        plt.xlabel('Time')
        plt.title('Speed time coherence')
        plt.legend()
        plt.savefig(os.path.join(self.parent_dir, 'pre_processing', self.session, 'sanity_check', 'speed', 'time_coherence_speed.png'))
        plt.clf()