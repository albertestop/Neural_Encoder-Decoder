from Clopath.tests.recons_class.src.validation import validate


def training_loop(n_epochs, device, optimizer, scheduler, model, loss_fn, train_loader, val_loader, test_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0

        for instances, targets in train_loader:
            instances = instances.to(device=device)
            targets = targets.to(device=device).unsqueeze(1)
            
            tr_outputs = model(instances)
            tr_loss = loss_fn(tr_outputs, targets)

            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
            loss_train += tr_loss.item()

        loss_train /= len(train_loader.dataset)

        scheduler.step()

        if epoch == 1 or epoch % 10 == 0:
            loss_validation = validate(device, model, val_loader, loss_fn)
            print(f"Epoch {epoch}, Training loss {loss_train:.4f},"
                f" Lr = {scheduler.get_last_lr()[0]},"
                f" Validation loss {loss_validation.item():.4f}")
    
    return model

