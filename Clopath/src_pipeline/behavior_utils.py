import numpy as np

class BehaviorPipeline():
    def __init__(self):
        pass


    def __call__(self, behavior, behavior_params):
        
        if behavior_params['speed'] == 'original':
            behavior[1, :] = behavior[1, :]

        elif behavior_params['speed'] == 'mean':
            behavior[1, :] = behavior[1, :].mean()
            
        elif behavior_params['speed'] == '0':
            behavior[1, :] = 0
        
        elif behavior_params['speed'] == 'max':
            behavior[1, :] = behavior[1, :].max()
        
        return behavior


