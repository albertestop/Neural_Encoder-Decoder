from pathlib import Path
import torch
import Clopath.src.utils_reconstruction as utils
import cv2

class Reconstructor:
    def __init__(self, device, number_models, mask_update_expanded, population_mask,
        reconstruct_params, epoch_switch, strides_all, gt_responses, 
        predictor_withgrads, inputs):

        self.subbatch_size = reconstruct_params['subbatch_size']
        self.n_steps = reconstruct_params['n_steps']
        self.use_adam = reconstruct_params['use_adam']
        self.lr = reconstruct_params['lr']
        self.lr_warmup_epochs = reconstruct_params['lr_warmup_epochs']
        self.with_gradnorm = reconstruct_params['with_gradnorm']
        self.clip_grad = reconstruct_params['clip_grad']
        self.adam_beta1 = reconstruct_params['adam_beta1']
        self.adam_beta2 = reconstruct_params['adam_beta2']
        self.pix_decay_rate = reconstruct_params['pix_decay_rate']
        self.input_noise = reconstruct_params['input_noise']
        self.epoch_switch = epoch_switch
        self.strides_all = strides_all
        self.device = device
        self.number_models = number_models
        self.mask_update_expanded = mask_update_expanded
        self.population_mask = population_mask
        self.gt_responses = gt_responses
        self.predictor_withgrads = predictor_withgrads
        self.inputs = inputs

    def compute_frames(self, subbatch, number_of_subbatches, subbatch_shift):
        """
        Computes the subbatch frames.
        """
        if subbatch == number_of_subbatches - 1:
            start_frame = self.inputs.shape[1] - self.subbatch_size
            end_frame = self.inputs.shape[1]
        else:
            start_frame = subbatch * subbatch_shift
            end_frame = subbatch * subbatch_shift + self.subbatch_size
        subbatch_frames = range(start_frame, end_frame)
        return subbatch_frames

    def comp_video_noise(self, video_pred, subbatch_frames):
        """
        If self.input_noise true we add noise to the video_prediction maybe to encourage exploration in the optimization
        """
        if self.input_noise > 0: 
            video_noise = torch.randn_like(video_pred[:, :, subbatch_frames, :, :]) * self.input_noise
        else:
            video_noise = torch.zeros_like(video_pred[:, :, subbatch_frames, :, :])
        
        return video_noise

    def concat_video_behaviour(self, video_input, behaviour_input):
        """
        Concatenate the predicted video with the behavior data without neural responses.
        """
        if behaviour_input.shape[1] == 1: # duplicate to match length of vid
            behaviour_input = behaviour_input.repeat(1,video_input.shape[1],1,1)
        return torch.cat((video_input, behaviour_input), axis=1).detach().requires_grad_(True)


    def prepare_inputs(self, video_pred, subbatch_frames):

        video_noise = self.comp_video_noise(
            video_pred, subbatch_frames
        )

        input_prediction = self.concat_video_behaviour(
            video_pred[:, :, subbatch_frames, :, :] + video_noise,
            self.inputs[None, 1:, subbatch_frames, :, :]
        )

        return input_prediction

    def compute_gradients(self, responses_predicted_new, subbatch_frames,
        input_prediction, gradnorm, n, subbatch, gradients_fullvid):

        loss = utils.response_loss_function(
            responses_predicted_new,
            self.gt_responses[:, subbatch_frames].clone().detach(),
            mask=self.population_mask
        )

        # Compute gradients to apply to input_prediction to reduce loss
        loss.backward()
        gradients = input_prediction.grad

        # Normalize and rescale gradients
        gradnorm[n, subbatch] = torch.norm(gradients) 
        if self.with_gradnorm:
            gradients = gradients / (gradnorm[n, subbatch] + 1e-6)
        else:
            gradients = gradients * 100

        # Store gradients only of the video
        gradients_fullvid[n, subbatch, :, subbatch_frames, :, :] = gradients[:, 0:1, :, :, :] 

        return gradients_fullvid, loss

    def compute_lr(self, i):
        if self.lr_warmup_epochs > 0 and i < self.lr_warmup_epochs:
            lr_current = self.lr * min(1, i / self.lr_warmup_epochs)
        else:
            lr_current = self.lr
        return lr_current


    def apply_grads(self, i, lr_current, gradients_fullvid, video_pred):
        if not self.use_adam:
            video_pred = torch.add(video_pred, -lr_current * gradients_fullvid[0:1, 0:1])
        else:
            if i == 1:
                self.m = torch.zeros_like(gradients_fullvid)
            lr_t = lr_current * (1 - self.adam_beta2 ** (i + 1)) ** 0.5 / (1 - self.adam_beta1 ** (i + 1))
            self.m = self.adam_beta1 * self.m + (1 - self.adam_beta1) * gradients_fullvid
            m_hat = self.m / (1 - self.adam_beta1 ** (i + 1))
            video_pred = torch.add(video_pred, -lr_t * m_hat)
        
        return video_pred

    def iterate(self, i, video_pred):
        
        # We take the stride corresponding to the epoch
        for n in range(self.n_steps):
            if i < self.epoch_switch[n]:
                subbatch_shift = self.strides_all[n]
                break

        number_of_subbatches = 2 + (self.inputs.shape[1] - self.subbatch_size) // subbatch_shift # we will divide the video in number_of_subbatches segments so that gradients can be computed on smaller chunks of frames

        # We create one gradient por each video pixel x model x subbatch. Pred shape = (1, ,1, 300, 36, 64), gradients shape = (models, subbatches, 300, 36, 64)
        gradients_fullvid = torch.zeros_like(video_pred).repeat(self.number_models, number_of_subbatches, 1, 1, 1, 1).to(self.device).requires_grad_(False)
        gradients_fullvid = gradients_fullvid.fill_(float('nan'))
        gradnorm = torch.zeros(self.number_models, number_of_subbatches).to(self.device)

        for n in range(self.number_models):

            for subbatch in range(number_of_subbatches):
                
                subbatch_frames = self.compute_frames(
                    subbatch, number_of_subbatches, subbatch_shift
                )

                input_prediction = self.prepare_inputs(
                    video_pred, subbatch_frames
                )

                responses_predicted_new = self.predictor_withgrads[n](input_prediction) # Compute prediction to reconstructing video

                gradients_fullvid, loss = self.compute_gradients(
                    responses_predicted_new, subbatch_frames, 
                    input_prediction, gradnorm, n, subbatch, gradients_fullvid
                )
                
        gradients_fullvid = torch.nanmean(gradients_fullvid, axis=1, keepdim=True) #join gradients to have a sinle gradient tensor for a full video
        gradients_fullvid = torch.nanmean(gradients_fullvid, axis=0, keepdim=False)
        gradients_fullvid = gradients_fullvid * self.mask_update_expanded #apply spatial mask to gradients//gradients to mask
        if self.clip_grad is not None:
            gradients_fullvid = torch.clip(gradients_fullvid, -1 * self.clip_grad, self.clip_grad)

        # Apply Gradients
        if i > 0:
            lr_current = self.compute_lr(i)
            video_pred = self.apply_grads(i, lr_current, gradients_fullvid, video_pred)

        video_pred = torch.clip(video_pred, 0, 255)

        if self.pix_decay_rate > 0:
            video_pred = ((video_pred - 255 / 2) * (1 - self.pix_decay_rate)) + (255 / 2)

        video_pred = video_pred.detach().requires_grad_(True) # we detach video_pred from the current computational graph so that new gradients are computed in the next iteration

        return video_pred, loss, gradients_fullvid


    def reconstruct_video(self, trial_save_path):

        # Leer el archivo TIFF usando el mismo nombre de archivo
        tif_path = Path(f'{trial_save_path}/optimized_input.tif')
        mp4_path = f'{trial_save_path}/optimized_input.mp4'

        # Asegurarse de que el archivo TIFF existe
        if not tif_path.exists():
            raise FileNotFoundError(f"The TIFF file {tif_path} does not exist.")

        tif_frames = cv2.imreadmulti(str(tif_path))[1]
        if not tif_frames:
            raise FileNotFoundError(f"No frames found in the TIFF file at {tif_path}. Please ensure the file exists and contains data.")

        height, width = tif_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output = cv2.VideoWriter(str(mp4_path), fourcc, 30, (width, height))

        for frame in tif_frames:
            video_output.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        video_output.release()

        return mp4_path