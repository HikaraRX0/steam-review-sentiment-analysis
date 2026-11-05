import torch
from sklearn.metrics import accuracy_score, f1_score

def _run_epoch(model, loader, loss_fn, optimizer, device):
    """Run one training or evaluation epoch"""
    training = optimizer is not None
    model.train(training) # switch to training model if training is True

    total_loss = 0.0
    targets = []  # y
    prediction = []  # y_hat

    with torch.set_grad_enabled(training):
        # load a mini batch each time
        for features, labels in loader:
            features = tuple(x.to(device) for x in features)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            # forward
            pred = model(*features)
            # compute loss
            loss = loss_fn(pred, labels)
            if training:
                # compute gradients and update parameters
                loss.backward()
                optimizer.step()

            # accumulate sample-weighted loss
            total_loss += loss.item() * labels.size(0)

            # collect true labels and prediction classes
            targets.extend(labels.detach().cpu().tolist())
            prediction.extend(pred.argmax(dim=1).detach().cpu().tolist())

        return {
            "loss": total_loss/len(targets),
            "accuracy": accuracy_score(targets, prediction),
            "macro_f1": f1_score(targets, prediction, average="macro", zero_division=0)
                }

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Run one training epoch."""
    return _run_epoch(model, loader,loss_fn, optimizer,device)


def evaluate(model, loader, loss_fn, device):
    """Evaluate without updating parameters"""
    return _run_epoch(model, loader, loss_fn, None, device)


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device):
    """Run training for multiple epochs."""
    metric_history = [] # record metrics scores: loss, accuracy, marco_f1

    for epoch in range(num_epochs):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        metric_history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        })

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train loss: {train_metrics['loss']:.3f} | "
            f"Train accuracy: {train_metrics['accuracy']:.3f} | "
            f"Val loss: {val_metrics['loss']:.3f} | "
            f"Val accuracy: {val_metrics['accuracy']:.3f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.3f}"
        )

    return metric_history






