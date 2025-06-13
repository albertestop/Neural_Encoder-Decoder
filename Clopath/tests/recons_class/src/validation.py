import torch


def validate(device, model, val_loader, loss_fn):
    with torch.no_grad():
        loss_validation = 0.0
        for instances, targets in val_loader:
            instances = instances.to(device=device)
            targets = targets.to(device=device).unsqueeze(1)
            outputs = model(instances)
            loss_val = loss_fn(outputs, targets)
            loss_validation += loss_val
        
        loss_validation /= len(val_loader.dataset)
        
        return loss_validation