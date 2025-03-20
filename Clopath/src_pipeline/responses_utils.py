import numpy as np
import torch

class ResponsesPipeline():
    def __init__(self, trial_index, device):
        self.trial_index = trial_index
        self.device = device

    def reduce_population(self, responses, population_reduction):
        """
        Performs ablation to neural responses.

        It takes the neural responses and sets a given % of them defined in 
        config_reconstruct to 0
        """
        if population_reduction > 0: 
            manual_seed = self.trial_index
            torch.manual_seed(manual_seed)
            responses = torch.tensor(responses).to(self.device)
            population_mask = torch.rand((responses.shape[0], 1), device=self.device) > population_reduction
            responses = responses * population_mask.repeat(1, responses.shape[1])
            responses = np.array(responses.cpu())
        else:
            population_mask = None

        return responses, population_mask
        

    def randomize_responses(self, responses):
        """
        Takes the neural responses and shuffles the neuron order
        """
        manual_seed = self.trial_index
        torch.manual_seed(manual_seed)
        return responses[torch.randperm(responses.shape[0]), :]


    def __call__(self, responses, responses_params):
        
        if responses_params['randomize_neurons']:
            responses = self.randomize_responses(responses)

        responses, population_mask = self.reduce_population(responses, responses_params['population_reduction'])
        
        return responses, population_mask


