import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import os
import sys
working_dir = os.path.join(os.curdir[:-9])
sys.path.append(working_dir)

sens_data_dir = '/home/albertestop/data/processed_data/sensorium_all_2023/dynamic29515'
lab_data_dir = '/home/albertestop/data/processed_data/sensorium_all_2023/2025-02-26_02_ESPM126_000'

trial = '26.npy'

sens_video = np.load(os.path.join(sens_data_dir, 'data', 'videos', trial))
lab_video = np.load(os.path.join(lab_data_dir, 'data', 'videos', trial))
sens_response = np.load(os.path.join(sens_data_dir, 'data', 'responses', trial))
lab_response = np.load(os.path.join(lab_data_dir, 'data', 'responses', trial))[:, ::9]
sens_behavior = np.load(os.path.join(sens_data_dir, 'data', 'behavior', trial))
lab_behavior = np.load(os.path.join(lab_data_dir, 'data', 'behavior', trial))
sens_pupil_center = np.load(os.path.join(sens_data_dir, 'data', 'pupil_center', trial))
lab_pupil_center = np.load(os.path.join(lab_data_dir, 'data', 'pupil_center', trial))

check_data = []

# Videos
if sens_video.shape == lab_video.shape: check_data.append('Video shapes are equal with shape = ' + str(sens_video.shape))
else: check_data.append('Video shapes are different: sens_video_shape = ' + str(sens_video.shape) + ', lab_video_shape = ' + str(lab_video.shape))
plt.figure(2)
plt.imshow(sens_video[:, :, 0], cmap='gray')
plt.title('Sensorium video first frame')
plt.axis('off')
plt.savefig('plot/data/data_study/First_frame_sens.png')
plt.figure(3)
plt.imshow(lab_video[:, :, 0], cmap='gray')
plt.title('Lab video first frame')
plt.axis('off')
plt.savefig('plot/data/data_study/First_frame_lab.png')

# Responses
if sens_response.shape == lab_response.shape: check_data.append('Response shapes are equal with shape = ' + str(sens_response.shape))
else: check_data.append('Response shapes are: sens_response_shape = ' + str(sens_response.shape) + ', lab_response_shape = ' + str(lab_response.shape))
plt.figure(4)
plt.plot(np.arange(sens_response.shape[1]), sens_response[38, :], label='Sens neuron response')
plt.plot(np.arange(lab_response.shape[1]), lab_response[112, :], label='Lab neuron response')
plt.xlabel('Frame')
plt.ylabel('Amplitude')
plt.title('Sensorium vs lab neuron response')
plt.legend()
plt.savefig('plot/data/data_study/Response_single_SensvsLab_ex1.png')

plt.figure(8)
plt.plot(np.arange(sens_response.shape[1]), sens_response[1000, :], label='Sens neuron response')
plt.plot(np.arange(lab_response.shape[1]), lab_response[1000, :], label='Lab neuron response')
plt.xlabel('Frame')
plt.ylabel('Amplitude')
plt.title('Sensorium vs lab neuron response')
plt.legend()
plt.savefig('plot/data/data_study/Response_single_SensvsLab_ex2.png')

plt.figure(5)
plt.imshow(sens_response, aspect='auto', cmap='viridis', vmin=0, vmax=50, interpolation='nearest')
plt.colorbar(label='Response Intensity')
plt.xlabel('Frame')
plt.ylabel('Neuron')
plt.title('Sens response')
plt.savefig('plot/data/data_study/Response_sens.png')

plt.figure(6)
plt.imshow(lab_response, aspect='auto', cmap='viridis', vmin=0, vmax=50, interpolation='nearest')
plt.colorbar(label='Response Intensity')
plt.xlabel('Frame')
plt.ylabel('Neuron')
plt.title('Lab response')
plt.savefig('plot/data/data_study/Response_lab.png')

# Behavior
if sens_behavior.shape == lab_behavior.shape: check_data.append('Behavior shapes are equal with shape = ' + str(sens_behavior.shape))
else: check_data.append('Behavior shapes are different: sens_behavior_shape = ' + str(sens_behavior.shape) + ', lab_behavior_shape = ' + str(lab_behavior.shape))
plt.figure(7)
frame_bright = 60*lab_video.mean(axis=(0,1))/250
plt.plot(np.arange(lab_behavior.shape[1]), frame_bright, label='Lab video frame brightness', color='green', alpha=0.5)
plt.plot(np.arange(sens_behavior.shape[1]), sens_behavior[0, :], label='Sens eye dilaiton', color='#ff7f0e')
plt.plot(np.arange(lab_behavior.shape[1]), lab_behavior[0, :], label='Lab eye dilation', color='#1f77b4')
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Pupil dilation')
plt.savefig('plot/data/data_study/EyeDilaiton_SensvsLab.png')

# Pupil Center
if sens_pupil_center.shape == lab_pupil_center.shape: check_data.append('Pupil center shapes are equal with shape = ' + str(lab_pupil_center.shape))
else: check_data.append('Pupil center shapes are different: lab_pupil_center_shape = ' + str(lab_pupil_center.shape) + ', sens_pupil_center_shape = ' + str(sens_pupil_center.shape))
plt.figure(9)
plt.plot(np.arange(lab_pupil_center.shape[1]), lab_pupil_center[0, :], label='Lab pupil center X')
plt.plot(np.arange(lab_pupil_center.shape[1]), lab_pupil_center[1, :], label='Lab pupil center Y')
plt.plot(np.arange(sens_pupil_center.shape[1]), sens_pupil_center[0, :], label='Sens pupil center X')
plt.plot(np.arange(sens_pupil_center.shape[1]), sens_pupil_center[1, :], label='Sens pupil center Y')
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Pupil position')
plt.savefig('plot/data/data_study/PupilPos_SensvsLab.png')

with open('plot/data/data_study/data_study.txt' , "w") as file:
    for item in check_data:
        file.write(f"{item}\n")
file.close()