import numpy as np
import os
import matplotlib.pyplot as plt



class PupilSC():
    def __init__(self, parent_dir, session):
        self.parent_dir = parent_dir
        self.session = session

    def pupil_dilation_time_coherence(self, trials_time, behavior):
        pupil_size = []
        for t_0 in trials_time:
            indexes = np.where((behavior.pupil_dilation_time > t_0 - 8) & (behavior.pupil_dilation_time < t_0 + 8))
            pupil_size.append(behavior.pupil_dilation[indexes])
        min_len = min([len(sublist) for sublist in pupil_size])
        pupil_size = [sublist[:min_len] for sublist in pupil_size]
        pupil_size_mean_tr = np.mean(pupil_size, axis=0)
        pupil_size_mean = np.mean(pupil_size_mean_tr)
        rec_rate = len(np.where((behavior.pupil_dilation_time >= 10) & (behavior.pupil_dilation_time < 11))[0])

        plt.clf()
        plt.plot(np.arange(-8, 8, (1/(len(pupil_size_mean_tr)) * 16)), pupil_size_mean_tr, label='pupil_dilation_mean')
        plt.axhline(y=pupil_size_mean, color='green', linestyle='--', label='mean session pupil dilaion')
        plt.axvline(x=0, color='red', linestyle='--', label='trial_start')
        plt.ylabel('Pupil_dilation')
        plt.xlabel('Time')
        plt.title('Pupil dilation time coherence')
        plt.legend()
        plt.savefig(os.path.join(self.parent_dir, 'pre_processing', self.session, 'sanity_check', 'pupil', 'time_coherence_pupil_dilation.png'))
        plt.clf()


    def pupil_pos_time_coherence(self, trials_time, pupil_pos):
        pupil_movmts = []
        x_gradient = np.gradient(pupil_pos.data[0, :])
        y_gradient = np.gradient(pupil_pos.data[1, :])
        eye_movement = np.sqrt(np.power(x_gradient, 2) + np.power(y_gradient, 2))
        for t_0 in trials_time:
            indexes = np.where((pupil_pos.time > t_0 - 8) & (pupil_pos.time < t_0 + 8))
            pupil_movmts.append(eye_movement[indexes])
        min_len = min([len(sublist) for sublist in pupil_movmts])
        pupil_movmts = [sublist[:min_len] for sublist in pupil_movmts]
        pupil_movmts_mean_tr = np.mean(pupil_movmts, axis=0)
        rec_rate = len(np.where((pupil_pos.time >= 10) & (pupil_pos.time < 11))[0])

        plt.clf()
        plt.plot(np.arange(-8, 8, (1/(len(pupil_movmts_mean_tr)) * 16)), pupil_movmts_mean_tr, label='eye_movement_mean')
        plt.axvline(x=0, color='red', linestyle='--', label='trial_start')
        plt.ylabel('Eye movement')
        plt.xlabel('Time')
        plt.title('Eye movement time coherence')
        plt.legend()
        plt.savefig(os.path.join(self.parent_dir, 'pre_processing', self.session, 'sanity_check', 'pupil', 'time_coherence_pupil_pos.png'))
        plt.clf()

    
    def pupil_dilation_sens_comp(self, sens_behavior, pupil_dilation, lab_video, pre_processing=True):
        frame_bright = 60*lab_video.mean(axis=(0,1))/250
        sens_tr_length = sens_behavior.shape[1]
        plt.clf()
        plt.plot(np.arange(sens_tr_length), frame_bright[:sens_tr_length], label='Lab video frame brightness', color='green', alpha=0.5)
        plt.plot(np.arange(sens_tr_length), sens_behavior[0, :], label='Sens eye dilaiton', color='#ff7f0e')
        plt.plot(np.arange(sens_tr_length), pupil_dilation[:sens_tr_length], label='Lab eye dilation', color='#1f77b4')
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Pupil dilation')
        plt.savefig(os.path.join(self.parent_dir, 'pre_processing', self.session, 'sanity_check', 'pupil', 'EyeDilaiton_SensvsLab.png'))
        plt.clf()

    def pupil_pos_sens_comp(self, sens_pupil_center, lab_pupil_center, pre_processing=True):
        if not pre_processing:
            check_data = []
            if sens_pupil_center.shape == lab_pupil_center.shape: check_data.append('Pupil center shapes are equal with shape = ' + str(lab_pupil_center.shape))
            else: check_data.append('Pupil center shapes are different: lab_pupil_center_shape = ' + str(lab_pupil_center.shape) + ', sens_pupil_center_shape = ' + str(sens_pupil_center.shape))
        sens_tr_length = sens_pupil_center.shape[1]
        plt.clf()
        plt.plot(np.arange(sens_tr_length), lab_pupil_center[0, :sens_tr_length], label='Lab pupil center X')
        plt.plot(np.arange(sens_tr_length), lab_pupil_center[1, :sens_tr_length], label='Lab pupil center Y')
        plt.plot(np.arange(sens_tr_length), sens_pupil_center[0, :], label='Sens pupil center X')
        plt.plot(np.arange(sens_tr_length), sens_pupil_center[1, :], label='Sens pupil center Y')
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Pupil position')
        plt.savefig(os.path.join(self.parent_dir, 'pre_processing', self.session, 'sanity_check', 'pupil', 'PupilPos_SensvsLab.png'))
        plt.clf()
