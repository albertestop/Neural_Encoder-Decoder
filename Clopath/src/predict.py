import torch
from src import constants

class Predict:
    
    def predict(predictor, device, video, behavior, pupil_center, mouse_index, video_length, stage='original'):
        print(f"\nDiagnostic - Model predictions of the {stage} video:")
        responses_predicted = torch.zeros((constants.num_neurons[mouse_index], video_length, len(predictor)), device=device)
        print(f"Response tensor shape: {responses_predicted.shape}")
        for n in range(len(predictor)):
            prediction = predictor[n].predict_trial(
                video=video,
                behavior=behavior,
                pupil_center=pupil_center,
                mouse_index=mouse_index
            )
            responses_predicted[:, :, n] = torch.from_numpy(prediction).to(device)
        print(f"Model predicted responses shape: {responses_predicted.shape}")

        return responses_predicted

    def process_predictions(responses, responses_predicted, population_mask, mask=True, mean=True):
        
        # Apply population mask
        if mask and population_mask is not None:
            mask_torch = population_mask  # Ya es un tensor en el dispositivo
            responses_predicted = responses_predicted * mask_torch[:, :, None]
            print("\nDiagnostic - Population mask details:")
            print(f"Model predicted responses shape after mask: {responses_predicted.shape}")

        # Calcular la media de todas las predicciones de los modelos + check shape mean_predictions = shape responses
        if mean: 
            responses_predicted = responses_predicted.mean(axis=2)  # Resultado: (8122, 300)
            if responses_predicted.shape != responses.shape:
                raise ValueError(f"Mismatch shapes: The mean of the model predictions {responses_predicted.shape} vs responses ground truth {responses.shape}")

        return responses_predicted
