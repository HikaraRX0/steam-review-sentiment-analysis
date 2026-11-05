import torch
from sklearn.metrics import accuracy_score, f1_score
from time import perf_counter
from utils import synchronize_device

def _run_epoch(model, loader, loss_fn, optimizer, device):
    """Run one training or evaluation epoch"""
    training = optimizer is not None
    model.train(training) # switch to training model if training is True

    total_loss = 0.0
    total_tokens = 0
    targets = []  # y
    prediction = []  # y_hat

    with torch.set_grad_enabled(training):
        # load a mini batch each time
        for features, labels in loader:
            total_tokens += features[1].sum().item()

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
            "macro_f1": f1_score(targets, prediction, average="macro", zero_division=0),
            "num_tokens": total_tokens,
                }

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Run one training epoch."""
    return _run_epoch(model, loader,loss_fn, optimizer,device)


def evaluate(model, loader, loss_fn, device):
    """Evaluate without updating parameters"""
    return _run_epoch(model, loader, loss_fn, None, device)


def train_model(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, device, on_epoch_end=None):
    """Run training for multiple epochs."""
    device = torch.device(device)  # support string
    metric_history = []  # record metrics scores: loss, accuracy, marco_f1

    device_name = str(device)
    if device.type == "cuda":
        device_name += f"({torch.cuda.get_device_name(device)})"

    print(f"Training on: {device_name}")
    
    for epoch in range(num_epochs):
        synchronize_device(device)
        start = perf_counter()

        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        synchronize_device(device)
        train_seconds = perf_counter() - start

        val_metrics = evaluate(model, val_loader, loss_fn, device)

        tokens_per_second = (
            train_metrics["num_tokens"] / train_seconds
        )

        metric_history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "train_seconds": train_seconds,
            "tokens_per_second": tokens_per_second,
        })

        if on_epoch_end is not None:
            on_epoch_end(metric_history)

    total_tokens = sum(
        item["train"]["num_tokens"] for item in metric_history
    )
    total_seconds = sum(
        item["train_seconds"] for item in metric_history
    )

    print(f"Average speed: {total_tokens / total_seconds:.0f} tokens/sec")

    return metric_history






