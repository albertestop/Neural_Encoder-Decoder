import matplotlib.pyplot as plt
import numpy as np
import torch
import shutil
from pathlib import Path

from src import constants
import Clopath.src.utils_reconstruction as utils
import Clopath.src.image_similarity as im_sim

class Evaluator():
    def __init__(self, device, mouse_index, responses_shape, video_length,
        epoch_switch, population_mask, number_models,
        mask_eval, trial_save_path, predictor, eval_params, 
        video, behavior, pupil_center, responses, 
        responses_predicted_mean, responses_predicted_gray, current_dir
        ):
        
        self.mouse_index = mouse_index
        self.device = device
        self.responses_shape = responses_shape
        self.epoch_switch = epoch_switch
        self.population_mask = population_mask
        self.plot_iter = eval_params['plot_iter']
        self.track_iter = eval_params['track_iter']
        self.eval_frame_skip = eval_params['eval_frame_skip']
        self.number_models = number_models
        self.mask_eval = mask_eval
        self.trial_save_path = trial_save_path
        self.video_length = video_length
        self.predictor = predictor
        self.video = video
        self.behavior = behavior
        self.pupil_center = pupil_center
        self.responses = responses
        self.current_dir = current_dir

        # Compute loss and corr between ground truth y predicted responses
        self.loss_gt = utils.response_loss_function(
            responses_predicted_mean,
            responses.clone().detach(),
            mask=population_mask)
        self.response_corr_gt = self.comp_responses_corr(
            responses, responses_predicted_mean
        )

        print(f'\nDiagnostic - Model response predictions')
        print(f'Ground truth test loss: {self.loss_gt.item()}')
        print(f'Response correlation ground truth: {self.response_corr_gt}')

        self.video_corr = []
        self.video_RMSE = []
        self.video_iter = []
        self.loss_all = []
        self.response_corr = []
        self.frame_corr = []  # Lista para almacenar la correlación frame a frame
        self.frame_RMSE = []  # Lista para almacenar el RMSE frame a frame
        
        self.fig, self.axs = plt.subplots(4, 4, figsize=(20, 20))
        self.fig.suptitle(f'{self.trial_save_path}', fontsize=16)
        self.axs[0, 0].imshow(np.concatenate((video[:, :, 0], video[:, :, -1]), axis=1), cmap='gray')
        self.axs[0, 0].axis('off')
        self.axs[0, 0].set_title('Frame 0 and ' + str(video.shape[2]))
        self.axs[1, 0].plot(behavior[:, :].T)
        self.axs[1, 0].set_title('Behavior')
        self.axs[2, 0].plot(pupil_center[:, :].T)
        self.axs[2, 0].set_title('Pupil position')
        self.axs[1, 1].imshow(responses.cpu().detach().numpy(), aspect='auto', vmin=0, vmax=10)
        self.axs[1, 1].set_title('Responses')
        self.axs[2, 1].imshow(responses_predicted_mean.cpu().detach().numpy(), aspect='auto', vmin=0, vmax=10)
        self.axs[2, 1].set_title('Predicted responses')
        self.axs[3, 0].imshow(responses_predicted_gray.mean(axis=2).cpu().detach().numpy(), aspect='auto', vmin=0, vmax=10)#----------------THIS ONE AND BELOW LINE COULD BE ADDED AS OPTIONAL TOGETHER WITH THE GRAY SCREEN RESPONSES SECTION
        self.axs[3, 0].set_title('Predicted responses gray screen')
        self.axs[0, 2].plot(np.sum(np.abs(np.diff(video, axis=2)), axis=(0, 1)))
        self.axs[0, 2].set_title('Video motion energy')
        self.fig.savefig(self.trial_save_path + '/results.png')


    def comp_responses_corr(self, responses_1, responses_2):
        """
        Computes the correlation between two high dimensional torch tensors.
        """
        return np.corrcoef(
        responses_1.cpu().detach().numpy().flatten(),
        responses_2.cpu().detach().numpy().flatten()
        )[0, 1]


    def update_figures(self, responses_predicted_full, gradients_fullvid,
        frame_corr_current, frame_RMSE_current, i):
        
        self.axs[0, 1].clear()
        self.axs[0, 1].imshow(np.concatenate((self.concat_video[0], self.concat_video[-1]), axis=1), cmap='gray', vmin=0, vmax=255)
        self.axs[0, 1].axis('off')
        self.axs[0, 1].set_title('First and last frame of ground truth and prediction')
        self.axs[3, 1].clear()
        self.axs[3, 1].imshow(responses_predicted_full.cpu().detach().numpy(), aspect='auto', vmin=0, vmax=10)
        self.axs[3, 1].set_title('Predicted responses reconstructed video')
        self.axs[0, 2].clear()
        self.axs[0, 2].plot(self.motion_energy_gt / np.max(self.motion_energy_gt), color='blue', label='Ground Truth')
        if i > 0:
            self.axs[0, 2].plot(self.motion_energy_recon / np.max(self.motion_energy_recon), color='red', label='Reconstruction')
        self.axs[0, 2].set_title('Motion Energy')
        self.axs[0, 2].legend()
        self.axs[0, 3].clear()
        self.axs[0, 3].imshow(gradients_fullvid[0, 0, self.eval_frame_skip:].mean(axis=(0)).cpu().detach().numpy(), vmin=-self.maxgrad / 5, vmax=self.maxgrad / 5)
        self.axs[0, 3].set_title(f'Mean gradient in space (vmax {np.round(self.maxgrad / 2, 4)})')
        self.axs[1, 2].clear()
        self.axs[1, 2].plot(self.video_iter, self.loss_all)
        self.axs[1, 2].axhline(y=self.loss_gt.item(), color='k', linestyle='--')
        self.axs[1, 2].set_title('Response loss')
        self.axs[1, 3].clear()
        self.axs[1, 3].plot(self.video_iter, self.response_corr)
        self.axs[1, 3].axhline(y=self.response_corr_gt, color='k', linestyle='--')
        self.axs[1, 3].set_title('Response correlation')
        self.axs[2, 2].clear()
        self.axs[2, 2].plot(self.video_iter, self.video_corr)
        self.axs[2, 2].set_title('Video correlation')
        self.axs[2, 3].clear()
        self.axs[2, 3].plot(self.video_iter, self.video_RMSE)
        self.axs[2, 3].set_title('Video RMSE')

        # Graficar correlación frame a frame
        if i == 0: mean_frame_corr = frame_corr_current[-1]
        else: mean_frame_corr = np.nanmean(frame_corr_current)  # np.nanmean ignora valores NaN
        self.axs[3, 2].clear()
        self.axs[3, 2].plot(frame_corr_current)
        self.axs[3, 2].axhline(y=mean_frame_corr, color='red', linestyle='--', label=f'Mean: {mean_frame_corr:.2f}')
        self.axs[3, 2].set_title('Frame-by-frame correlation')
        self.axs[3, 2].legend()

        # Graficar RMSE frame a frame
        if i == 0: mean_frame_RMSE = frame_RMSE_current[-1]
        else: mean_frame_RMSE = np.nanmean(frame_RMSE_current)  # np.nanmean ignora valores NaN
        self.axs[3, 3].clear()
        self.axs[3, 3].plot(frame_RMSE_current)
        self.axs[3, 3].axhline(y=mean_frame_RMSE, color='red', linestyle='--', label=f'Mean: {mean_frame_RMSE:.2f}')
        self.axs[3, 3].set_title('Frame-by-frame RMSE')
        self.axs[3, 3].legend()

        self.fig.savefig(f'{self.trial_save_path}/results.png')



    def iterate(self, i, loss, progress_bar, video_pred, gradients_fullvid):

        if i == 0:
            self.loss_init = loss.item()
            self.maxgrad = np.max([
                np.abs(gradients_fullvid[0, 0].mean(axis=(0)).cpu().detach().numpy().min()),
                np.abs(gradients_fullvid[0, 0].mean(axis=(0)).cpu().detach().numpy().max())
            ])

        if i == 0 or i % self.track_iter == 0 or i == self.epoch_switch[-1] - 1:
            progress_bar.set_postfix(variable_message=f'loss: {loss.item():.3f} / {self.loss_init:.0f}', refresh=True)
            progress_bar.update()
            ground_truth = np.moveaxis(self.video, [2], [0])
            reconstruction = video_pred[0, 0].cpu().detach().numpy()
            reconstruction = reconstruction[:, 14:14 + 36, :] #--------------------------------------------------------------we take the video prediction and remove the above and below 16 pixels added to apply the mask
            mask_cropped = self.mask_eval[14:14 + 36, :].cpu().detach().numpy()
            reconstruction_masked = reconstruction * mask_cropped + np.ones_like(reconstruction) * (1 - mask_cropped) * 255 / 2

            responses_predicted_full = torch.zeros((constants.num_neurons[self.mouse_index], self.video_length, self.number_models), device=self.device) # we recompute the predicted responses with the new videos
            for n in range(len(self.predictor)):
                prediction = self.predictor[n].predict_trial(
                    video=np.moveaxis(reconstruction_masked, [0], [2]),
                    behavior=self.behavior,
                    pupil_center=self.pupil_center,
                    mouse_index=self.mouse_index
                )
                responses_predicted_full[:, :, n] = torch.from_numpy(prediction).to(self.device)

            responses_predicted_full = responses_predicted_full.mean(axis=2)
            
            if self.population_mask is not None:
                responses_predicted_full = responses_predicted_full * self.population_mask

            # Guardar el archivo TIFF con el nombre actualizado
            self.concat_video = utils.save_tif(
                ground_truth,
                reconstruction,
                f'{self.trial_save_path}/optimized_input.tif',
                mask=mask_cropped
            )
            self.ground_truth = ground_truth
            
            self.video_corr.append(im_sim.reconstruction_video_corr(
                ground_truth[self.eval_frame_skip:], reconstruction[self.eval_frame_skip:], mask_cropped))
            self.video_RMSE.append(im_sim.reconstruction_video_RMSE(
                ground_truth[self.eval_frame_skip:], reconstruction[self.eval_frame_skip:], mask_cropped))
            self.video_iter.append(i)
            
            # Loss and correlation of new responses vs gr truth responses
            loss_full = utils.response_loss_function(
                responses_predicted_full[:, self.eval_frame_skip:],
                self.responses[:, self.eval_frame_skip:].clone().detach(),
                mask=self.population_mask
            )
            self.loss_all.append(loss_full.item())

            response_corr_value = self.comp_responses_corr(
                self.responses, responses_predicted_full
            )
            self.response_corr.append(response_corr_value)

            print(f"Full loss at iteration {i}: {loss_full.item()}")
            print(f"Response correlation at iteration {i}: {response_corr_value}")

            self.motion_energy_gt = im_sim.video_energy(ground_truth[self.eval_frame_skip:], mask_cropped)
            self.motion_energy_recon = im_sim.video_energy(reconstruction[self.eval_frame_skip:], mask_cropped)

            # Cálculo de correlación y RMSE frame a frame
            frame_corr_current = []
            frame_RMSE_current = []
            for t in range(self.eval_frame_skip, ground_truth.shape[0]):
                gt_frame = ground_truth[t]
                recon_frame = reconstruction[t]

                # Aplicar máscara para seleccionar píxeles dentro de la región de interés
                valid_pixels = mask_cropped > 0
                gt_flat = gt_frame[valid_pixels].flatten()
                recon_flat = recon_frame[valid_pixels].flatten()

                # Correlación frame a frame
                if np.std(gt_flat) > 0 and np.std(recon_flat) > 0:
                    corr = np.corrcoef(gt_flat, recon_flat)[0, 1]
                else:
                    corr = np.nan  # Si la desviación estándar es cero, asignar NaN
                frame_corr_current.append(corr)

                # RMSE frame a frame
                rmse = np.sqrt(np.mean((gt_flat - recon_flat) ** 2))
                frame_RMSE_current.append(rmse)

            self.frame_corr.append(frame_corr_current)
            self.frame_RMSE.append(frame_RMSE_current)

            if i == 0 or i % self.plot_iter == 0 or i == self.epoch_switch[-1] - 1:

                self.update_figures(responses_predicted_full, gradients_fullvid,
                    frame_corr_current, frame_RMSE_current, i)




    def save_results(self, strides_all, trial, mask, training_time, video_pred):
        shutil.copy(self.current_dir / Path('config.py'), self.trial_save_path + '/config.py')

        recon_dict = {
            'epochs': self.epoch_switch[-1],
            'strides': strides_all,
            'strides_switch': self.epoch_switch,
            'track_iter': self.track_iter,
            'plot_iter': self.plot_iter,
            'mouse_index': self.mouse_index,
            'trial': trial,
            'eval_frame_skip': self.eval_frame_skip,
            'video_length': self.video_length,
            'population_mask': self.population_mask.cpu().numpy() if self.population_mask is not None else None,
            'mask': mask,
            'device': str(self.device),
            'video_iter': self.video_iter,
            'response_loss_gt': self.loss_gt.item(),
            'response_loss_full': self.loss_all,
            'response_corr_gt': self.response_corr_gt,
            'response_corr_full': self.response_corr,
            'video_corr': self.video_corr,
            'video_RMSE': self.video_RMSE,
            'frame_corr': self.frame_corr,
            'frame_RMSE': self.frame_RMSE,
            'motion_energy_gt': self.motion_energy_gt,
            'motion_energy_recon': self.motion_energy_recon,
            'training_time': training_time
        }
        np.save( f'{self.trial_save_path}/reconstruction_summary.npy', recon_dict)

        np.save(f'{self.trial_save_path}/reconstruction_array.npy', video_pred)
        
        plt.close(self.fig)

