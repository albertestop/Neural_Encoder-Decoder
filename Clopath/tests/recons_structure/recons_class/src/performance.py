import torch
import numpy as np

def performance(model, test_loader, device):
    model.eval()
    predictions, targets = torch.zeros((len(test_loader))), torch.zeros((len(test_loader)))
    with torch.no_grad():
        for i, (instance, target) in enumerate(test_loader):
            instance = instance.to(device=device)
            target = target.to(device=device)
            predictions[i] = model(instance)
            targets[i] = target

    predictions = np.array(predictions.cpu())
    targets = np.array(targets.cpu())

    print('\nModel study')
    print('\nPredictions vs targets:')
    idx = np.random.randint(0, len(predictions), size=5)
    print(predictions[idx], targets[idx])
    print('\nPrediction mean value:')
    print(np.mean(predictions))
    print('\nAccuracy:')
    predictions[predictions > 0.5] = 1
    predictions[predictions < 0.5] = 0
    acc = (predictions == targets).sum()/len(predictions)
    print(acc)

    


