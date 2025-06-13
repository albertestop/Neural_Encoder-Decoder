import matplotlib.pyplot as plt
import numpy as np

class Evaluator:
    def __init__(self, session, session_per, segment_length):

        self.out_path = f'/home/albertestop/Sensorium/Clopath/tests/recons_class/data/{session}.png'
        self.prob = np.array([0])
        self.x_prob = np.array([0])
        self.segment_length = segment_length
        self.session_per = session_per

        self.fig, self.ax = plt.subplots(figsize=(50, 10))


    def iterate(self, t_i, prediction, label=None):
        self.ax.clear()

        t_start = self.session_per[0, 0]
        t_end   = self.session_per[-1, 0] + self.session_per[-1, 1]
        self.ax.set_xlim(t_start, t_end)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Reconstruction Prob')
        if self.ax.get_legend_handles_labels()[1]:
            self.ax.legend()
        self.ax.grid(True)
        self.fig.tight_layout()

        for t0, duration in self.session_per:
            self.ax.axvspan(t0, t0 + duration,
                            facecolor='lightcoral', alpha=0.5)
        
        x = np.arange(t_i, t_i + self.segment_length, 1/30)
        y = np.repeat(prediction, x.size)
        self.prob = np.concatenate([self.prob, y])
        self.x_prob = np.concatenate([self.x_prob, x])
        self.ax.plot(self.x_prob, self.prob, label=label)
        self.fig.savefig(self.out_path)


    def save(self):
        self.fig.savefig(self.out_path)
        print(f"Saved figure to: {self.out_path}")