import numpy as np
import torch

def generate_instances(dataset):
    inputs = dataset[:, :-1]
    targets = dataset[:, -1]
    return inputs, targets

def split_dataset(instances, targets, fraction=0.2):

    n_samples = instances.shape[0]
    n_val = int(fraction * n_samples)
    shuffled_indices = torch.randperm(n_samples)
    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]

    train_instances = instances[train_indices]
    train_targets = targets[train_indices]
    val_instances = instances[val_indices]
    val_targets = targets[val_indices]
    
    return train_instances, train_targets, val_instances, val_targets