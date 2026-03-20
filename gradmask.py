"""
GradMask Implementation for Federated Learning Backdoor Attacks

This module implements the gradient masking technique from Neurotoxin paper.
The core idea is to find parameters with small gradients on benign data,
and only allow backdoor updates on those parameters to preserve main task accuracy.
"""

import torch
import numpy as np


def compute_grad_mask(model, benign_data_loader, loss_func, device, ratio=0.95, aggregate_all_layer=False):
    """
    Compute gradient mask based on benign data.

    Args:
        model: The model to compute gradients for
        benign_data_loader: DataLoader with benign data
        loss_func: Loss function
        device: Device (cuda/cpu)
        ratio: Fraction of parameters to retain (e.g., 0.95 = keep 95% smallest gradients)
        aggregate_all_layer: If True, select top-k across all layers; if False, select per-layer

    Returns:
        mask_grad_list: List of mask tensors for each parameter
    """
    model.train()
    model.zero_grad()

    # Step 1: Compute gradients on benign data
    for batch_graphs, batch_labels in benign_data_loader:
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(torch.long).to(device)

        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)
        loss.backward(retain_graph=True)

    # Step 2: Generate masks based on gradient magnitudes
    mask_grad_list = []

    if aggregate_all_layer:
        # Select top-k across all layers
        grad_list = []
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_list.append(param.grad.abs().view(-1))

        grad_list = torch.cat(grad_list).to(device)
        k = int(len(grad_list) * ratio)
        _, indices = torch.topk(-1 * grad_list, k)  # -1 * grad_list to get smallest

        # Map indices back to individual parameters
        indices = indices.cpu().numpy()
        count = 0
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_size = len(param.grad.abs().view(-1))
                count_list = list(range(count, count + param_size))
                index_list = list(set(count_list).intersection(set(indices)))

                mask_flat = np.zeros(count + param_size)
                mask_flat[index_list] = 1.0
                mask_flat = mask_flat[count:count + param_size]
                mask = mask_flat.reshape(param.grad.abs().size())

                mask = torch.from_numpy(mask).float().to(device)
                mask_grad_list.append(mask)
                count += param_size
    else:
        # Select per-layer
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients = param.grad.abs().view(-1)
                gradients_length = len(gradients)
                k = int(gradients_length * ratio)
                _, indices = torch.topk(-1 * gradients, k)

                mask_flat = torch.zeros(gradients_length)
                mask_flat[indices.cpu()] = 1.0
                mask_grad_list.append(mask_flat.reshape(param.grad.size()).to(device))

    # Clear gradients
    model.zero_grad()

    return mask_grad_list


def apply_grad_mask(model, mask_grad_list):
    """
    Apply gradient mask to model gradients.

    Args:
        model: The model whose gradients will be masked
        mask_grad_list: List of mask tensors from compute_grad_mask
    """
    mask_iter = iter(mask_grad_list)
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            mask = next(mask_iter)
            param.grad = param.grad * mask
