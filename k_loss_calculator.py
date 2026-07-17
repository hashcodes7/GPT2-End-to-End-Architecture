import torch
import torch.nn.functional as F


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)

    return F.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    total_loss = 0.0

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i >= num_batches:
                break

            total_loss += calc_loss_batch(
                input_batch,
                target_batch,
                model,
                device,
            ).item()

    return total_loss / num_batches