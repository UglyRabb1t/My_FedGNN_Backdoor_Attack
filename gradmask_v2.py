"""
GradMask V2: Layer-Specific Gradient Masking for Federated Learning Backdoor Attacks

This module implements layer-specific gradient masking based on benign data gradients.
Key improvement: Different layers can have different retention ratios based on their gradient characteristics.

Core idea:
- Parameters with small gradients on benign data are good candidates for backdoor updates
- Use higher ratio for layers with smaller gradients (preserve more for backdoor)
- Use lower ratio for layers with larger gradients (filter more as they would be blocked by r=99)
"""

import torch
import numpy as np


def compute_grad_mask(model, benign_data_loader, loss_func, device, ratio=0.95, aggregate_all_layer=False,
                      layer_ratios=None, model_name='GCN', dataset='NCI1'):
    """
    Compute gradient mask based on benign data with layer-specific ratios.

    Args:
        model: The model to compute gradients for
        benign_data_loader: DataLoader with benign data
        loss_func: Loss function
        device: Device (cuda/cpu)
        ratio: Default fraction of parameters to retain (used if layer_ratios is None)
        aggregate_all_layer: If True, select top-k across all layers; if False, select per-layer
        layer_ratios: Dict mapping layer patterns to specific ratios. If None, use uniform ratio.
                      Example: {"layers.3.*": 0.9995, "layers.2.*": 0.999, ...}
        model_name: Model name for debugging
        dataset: Dataset name for debugging

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

    # Get layer-specific ratio for each parameter
    def get_layer_ratio(param_name, default_ratio):
        """Get the ratio for a specific parameter based on layer_ratios."""
        if layer_ratios is None:
            return default_ratio

        # Check for exact match first
        if param_name in layer_ratios:
            return layer_ratios[param_name]

        # Check for wildcard patterns
        for pattern, pattern_ratio in layer_ratios.items():
            if pattern.endswith('*'):
                # Remove the '*' and check if param_name starts with this pattern
                prefix = pattern[:-1]
                if param_name.startswith(prefix):
                    return pattern_ratio

        return default_ratio

    if aggregate_all_layer:
        # Select top-k across all layers (original behavior)
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
        # Select per-layer with layer-specific ratios
        param_names = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param_names.append(name)

        for idx, (name, param) in enumerate(model.named_parameters()):
            if param.requires_grad and param.grad is not None:
                # Get layer-specific ratio
                layer_ratio = get_layer_ratio(name, default_ratio=ratio)

                # Clamp ratio to [0, 1]
                layer_ratio = max(0.0, min(1.0, layer_ratio))

                gradients = param.grad.abs().view(-1)
                gradients_length = len(gradients)
                k = int(gradients_length * layer_ratio)

                if k > 0:
                    _, indices = torch.topk(-1 * gradients, k)
                    mask_flat = torch.zeros(gradients_length)
                    mask_flat[indices.cpu()] = 1.0
                    mask = mask_flat.reshape(param.grad.size()).to(device)
                else:
                    # If k=0, mask everything out
                    mask = torch.zeros_like(param.grad)

                mask_grad_list.append(mask)

                # Debug: print mask info
                kept_pct = (mask.sum().item() / mask.numel()) * 100
                print(f"  {name}: ratio={layer_ratio:.4f}, kept={kept_pct:.2f}% ({int(mask.sum().item())}/{mask.numel()})")

    # Clear gradients
    model.zero_grad()

    return mask_grad_list


def get_r99_layer_ratios():
    """
    Get layer ratios optimized for r=99 (retain 99% of smallest gradients).

    Design principle:
    - Layers with SMALL gradients (good for backdoor) -> HIGHER ratio
    - Layers with LARGE gradients (should be filtered) -> LOWER ratio

    Gradient analysis results from GCN + NCI1 (seed=0, 30 warmup epochs):
    - layers.3: 0.001154 (smallest) -> 0.9995
    - layers.2: 0.002041 -> 0.999
    - layers.1: 0.003342 -> 0.998
    - layers.0: 0.004586 -> 0.997
    - MLP output layers: 0.029-0.342 (largest) -> 0.950-0.985
    - embedding_h.bias: 0.039903 -> 0.970
    """
    return {
        # 【梯度极小的GNN层】保留更多用于后门
        "layers.3.apply_mod.linear.weight": 0.9995,  # 梯度最小，屏蔽0.05%
        "layers.3.apply_mod.linear.bias": 0.9995,
        "layers.2.apply_mod.linear.weight": 0.999,    # 屏蔽0.1%
        "layers.2.apply_mod.linear.bias": 0.999,
        "layers.1.apply_mod.linear.weight": 0.998,    # 屏蔽0.2%
        "layers.1.apply_mod.linear.bias": 0.998,

        # 【中等梯度GNN层】
        "layers.0.apply_mod.linear.weight": 0.997,    # 屏蔽0.3%
        "layers.0.apply_mod.linear.bias": 0.997,

        # 【输出层】梯度大，必须多屏蔽
        "MLP_layer.FC_layers.0.weight": 0.985,      # 屏蔽1.5%
        "MLP_layer.FC_layers.0.bias": 0.985,
        "MLP_layer.FC_layers.1.weight": 0.980,      # 屏蔽2.0%
        "MLP_layer.FC_layers.1.bias": 0.980,
        "MLP_layer.FC_layers.2.weight": 0.950,      # 屏蔽5.0%（梯度最大！）
        "MLP_layer.FC_layers.2.bias": 0.950,

        # 【嵌入层】
        "embedding_h.weight": 0.997,                 # 屏蔽0.3%
        "embedding_h.bias": 0.970,                  # 梯度较大，屏蔽3.0%

        # 【BatchNorm层】
        "layers.0.batchnorm_h.weight": 0.995,       # 屏蔽0.5%
        "layers.0.batchnorm_h.bias": 0.995,
        "layers.1.batchnorm_h.weight": 0.995,
        "layers.1.batchnorm_h.bias": 0.995,
        "layers.2.batchnorm_h.weight": 0.995,
        "layers.2.batchnorm_h.bias": 0.995,
        "layers.3.batchnorm_h.weight": 0.995,
        "layers.3.batchnorm_h.bias": 0.995,
    }


def get_r99_layer_ratios_simple():
    """
    Simplified layer ratios for r=99 (easier to implement and maintain).

    Uses wildcard patterns for layer grouping.
    """
    return {
        "layers.3.*": 0.9995,   # 梯度最小，几乎全保留
        "layers.2.*": 0.999,
        "layers.1.*": 0.998,
        "layers.0.*": 0.997,
        "MLP_layer.FC_layers.2.*": 0.950,  # 梯度最大，屏蔽5%
        "MLP_layer.FC_layers.1.*": 0.980,
        "MLP_layer.FC_layers.0.*": 0.985,
        "embedding_h.bias": 0.970,
        "embedding_h.weight": 0.997,
        "*.batchnorm_h.*": 0.995,
    }


def get_r95_layer_ratios():
    """
    Get layer ratios optimized for r=95 (retain 95% of smallest gradients).

    More aggressive masking compared to r=99.
    """
    return {
        # 【梯度极小的GNN层】
        "layers.3.apply_mod.linear.weight": 0.995,
        "layers.3.apply_mod.linear.bias": 0.995,
        "layers.2.apply_mod.linear.weight": 0.990,
        "layers.2.apply_mod.linear.bias": 0.990,
        "layers.1.apply_mod.linear.weight": 0.980,
        "layers.1.apply_mod.linear.bias": 0.980,

        # 【中等梯度GNN层】
        "layers.0.apply_mod.linear.weight": 0.970,
        "layers.0.apply_mod.linear.bias": 0.970,

        # 【输出层】梯度大，大量屏蔽
        "MLP_layer.FC_layers.0.weight": 0.920,
        "MLP_layer.FC_layers.0.bias": 0.920,
        "MLP_layer.FC_layers.1.weight": 0.900,
        "MLP_layer.FC_layers.1.bias": 0.900,
        "MLP_layer.FC_layers.2.weight": 0.850,  # 梯度最大！
        "MLP_layer.FC_layers.2.bias": 0.850,

        # 【嵌入层】
        "embedding_h.weight": 0.970,
        "embedding_h.bias": 0.940,

        # 【BatchNorm层】
        "*.batchnorm_h.weight": 0.960,
        "*.batchnorm_h.bias": 0.960,
    }


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
