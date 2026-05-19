"""
GradMask V2: Layer-Specific Gradient Masking for Federated Learning Backdoor Attacks

This module implements layer-specific gradient masking based on benign data gradients.
Key improvement: Different layers can have different retention ratios based on their gradient characteristics.

Core idea:
- Parameters with small gradients on benign data are good candidates for backdoor updates
- Use higher ratio for layers with smaller gradients (preserve more for backdoor)
- Use lower ratio for layers with larger gradients (filter more as they would be blocked by r=99)

Layer Presets (numbered for easy reference):
    Preset ID | Description
    ---------- | -------------------------------------------------
         1     | r99 with precise layer ratios (parameter-level)
         2     | r99 with simplified layer ratios (wildcard patterns)
         3     | r95 with aggressive layer ratios
         4     | r99 with gradient-proportional ratios (data-driven)
         5     | r99 balanced (GNN=0.995, MLP optimized) - 提高MLP保留率以提升CleanAcc
         6     | r99 aggressive (further reduce MLP) - 过度降低MLP保留率，效果差
         7     | r99 GNN微调组1 (GNN=0.99) - 复现L4的GNN设置，作为对照
         8     | r99 GNN微调组2 (GNN=0.995) - 略微提高GNN保留率
         9     | r99 GNN微调组3 (GNN=0.999) - 接近完全保留GNN层
         10    | r99 GNN微调组4 (GNN=1.0) - 完全保留GNN层，探索上限
         11    | r99 BatchNorm微调组1 (BatchNorm=0.97) - 降低BatchNorm保留率
         12    | r99 BatchNorm微调组2 (BatchNorm=0.995) - BatchNorm介于0.97和0.99之间
         13    | r99 BatchNorm微调组3 (BatchNorm=1.0) - 完全保留BatchNorm层
"""

import torch
import numpy as np


# Layer preset configurations
LAYER_PRESETS = {
    1: {
        'description': 'r99 precise ratios',
        'ratios': {
            # 【梯度极小的GNN层】保留更多用于后门
            "layers.3.apply_mod.linear.weight": 0.9995,
            "layers.3.apply_mod.linear.bias": 0.9995,
            "layers.2.apply_mod.linear.weight": 0.999,
            "layers.2.apply_mod.linear.bias": 0.999,
            "layers.1.apply_mod.linear.weight": 0.998,
            "layers.1.apply_mod.linear.bias": 0.998,

            # 【中等梯度GNN层】
            "layers.0.apply_mod.linear.weight": 0.997,
            "layers.0.apply_mod.linear.bias": 0.997,

            # 【输出层】梯度大，必须多屏蔽
            "MLP_layer.FC_layers.0.weight": 0.985,
            "MLP_layer.FC_layers.0.bias": 0.985,
            "MLP_layer.FC_layers.1.weight": 0.980,
            "MLP_layer.FC_layers.1.bias": 0.980,
            "MLP_layer.FC_layers.2.weight": 0.950,
            "MLP_layer.FC_layers.2.bias": 0.950,

            # 【嵌入层】
            "embedding_h.weight": 0.997,
            "embedding_h.bias": 0.970,

            # 【BatchNorm层】
            "layers.0.batchnorm_h.weight": 0.995,
            "layers.0.batchnorm_h.bias": 0.995,
            "layers.1.batchnorm_h.weight": 0.995,
            "layers.1.batchnorm_h.bias": 0.995,
            "layers.2.batchnorm_h.weight": 0.995,
            "layers.2.batchnorm_h.bias": 0.995,
            "layers.3.batchnorm_h.weight": 0.995,
            "layers.3.batchnorm_h.bias": 0.995,
        }
    },
    2: {
        'description': 'r99 simplified ratios',
        'ratios': {
            "layers.3.*": 0.9995,
            "layers.2.*": 0.999,
            "layers.1.*": 0.998,
            "layers.0.*": 0.997,
            "MLP_layer.FC_layers.2.*": 0.950,
            "MLP_layer.FC_layers.1.*": 0.980,
            "MLP_layer.FC_layers.0.*": 0.985,
            "embedding_h.bias": 0.970,
            "embedding_h.weight": 0.997,
            "*.batchnorm_h.*": 0.995,
        }
    },
    3: {
        'description': 'r95 aggressive ratios',
        'ratios': {
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
            "MLP_layer.FC_layers.2.weight": 0.850,
            "MLP_layer.FC_layers.2.bias": 0.850,

            # 【嵌入层】
            "embedding_h.weight": 0.970,
            "embedding_h.bias": 0.940,

            # 【BatchNorm层】
            "*.batchnorm_h.weight": 0.960,
            "*.batchnorm_h.bias": 0.960,
        }
    },
    4: {
        'description': 'r99 data-driven (gradient-proportional)',
        'ratios': {
            # 【GNN层偏置】梯度极小（1e-9到1e-8），几乎不更新，r=99会屏蔽它们
            # 但这些是后门的关键位置，必须几乎完全保留！
            "layers.3.apply_mod.linear.bias": 1.0,    # 4.49e-10
            "layers.2.apply_mod.linear.bias": 1.0,    # 7.46e-10
            "layers.1.apply_mod.linear.bias": 1.0,    # 1.75e-09
            "layers.0.apply_mod.linear.bias": 0.9999, # 3.36e-08

            # 【GNN深层权重】梯度非常小（0.002-0.004），关键层
            "layers.3.apply_mod.linear.weight": 0.9995, # 0.002308
            "layers.2.apply_mod.linear.weight": 0.999,  # 0.004082

            # 【嵌入层权重】梯度小
            "embedding_h.weight": 0.998, # 0.004306

            # 【GNN中间层权重】梯度小
            "layers.1.apply_mod.linear.weight": 0.997, # 0.006684

            # 【GNN浅层权重】梯度接近中位数
            "layers.0.apply_mod.linear.weight": 0.990, # 0.009172

            # 【BatchNorm层】梯度中等偏小
            "*.batchnorm_h.weight": 0.990,
            "*.batchnorm_h.bias": 0.990,

            # 【嵌入层偏置】梯度较大（0.04）
            "embedding_h.bias": 0.970, # 0.039903

            # 【MLP第一层】梯度较大（0.02-0.04）
            "MLP_layer.FC_layers.0.weight": 0.965, # 0.036180
            "MLP_layer.FC_layers.0.bias": 0.965,   # 0.023751

            # 【MLP第二层】梯度大（0.02-0.05）
            "MLP_layer.FC_layers.1.weight": 0.940, # 0.024840
            "MLP_layer.FC_layers.1.bias": 0.940,   # 0.044991

            # 【MLP输出层】梯度极大（0.2-0.5），r=99会屏蔽大量，这里应该更多屏蔽
            "MLP_layer.FC_layers.2.weight": 0.860, # 0.187284
            "MLP_layer.FC_layers.2.bias": 0.850,   # 0.496353
        }
    },
    5: {
        'description': 'r99 balanced (GNN=0.995, MLP optimized)',
        'ratios': {
            # GNN层：从1.0降到0.995，uniform r=99下本就保留得很好
            "layers.3.*": 0.995,
            "layers.2.*": 0.995,
            "layers.1.*": 0.995,
            "layers.0.*": 0.995,

            # MLP层：适度提高保留率，在保持攻击效果的同时提升CleanAcc
            "MLP_layer.FC_layers.2.*": 0.88,  # 输出层：从0.85提高到0.88
            "MLP_layer.FC_layers.1.*": 0.95,  # 中间层：从0.94提高到0.95
            "MLP_layer.FC_layers.0.*": 0.97,  # 输入层：从0.96提高到0.97

            "embedding_h.bias": 0.97,
            "embedding_h.weight": 0.99,
            "*.batchnorm_h.*": 0.99,
        }
    },
    6: {
        'description': 'r99 aggressive (further reduce MLP, protect GNN)',
        'ratios': {
            # GNN层：完全保留，因为它们是后门植入的关键
            "layers.3.*": 1.0,
            "layers.2.*": 1.0,
            "layers.1.*": 1.0,
            "layers.0.*": 1.0,

            # MLP层：进一步降低保留率，屏蔽更多良性梯度
            "MLP_layer.FC_layers.2.*": 0.80,  # 输出层：从0.86降到0.80，梯度极大必须多屏蔽
            "MLP_layer.FC_layers.1.*": 0.90,  # 中间层：从0.94降到0.90
            "MLP_layer.FC_layers.0.*": 0.93,  # 输入层：从0.965降到0.93

            "embedding_h.bias": 0.97,
            "embedding_h.weight": 0.99,
            "*.batchnorm_h.*": 0.99,
        }
    },
    7: {
        'description': 'r99 GNN微调组1 (GNN=0.99，L4对照)',
        'ratios': {
            # 目的：复现L4的GNN层设置，验证MLP层保持不变时GNN层0.99的效果
            # 期望：与L4结果相近，作为对照基准
            "layers.3.*": 0.99,
            "layers.2.*": 0.99,
            "layers.1.*": 0.99,
            "layers.0.*": 0.99,

            # MLP层：保持L4的比例（已被证明是较优配置）
            "MLP_layer.FC_layers.2.*": 0.86,  # 输出层：梯度极大(0.2-0.5)
            "MLP_layer.FC_layers.1.*": 0.94,  # 中间层：梯度较大(0.02-0.05)
            "MLP_layer.FC_layers.0.*": 0.965, # 输入层：梯度较大(0.02-0.04)

            "embedding_h.bias": 0.97,
            "embedding_h.weight": 0.998,
            "*.batchnorm_h.*": 0.99,
        }
    },
    8: {
        'description': 'r99 GNN微调组2 (GNN=0.995，略微提高)',
        'ratios': {
            # 目的：测试GNN层保留率从0.99略微提高到0.995的效果
            # 期望：可能略微提升ASR持久性，因为GNN层是后门关键
            "layers.3.*": 0.995,
            "layers.2.*": 0.995,
            "layers.1.*": 0.995,
            "layers.0.*": 0.995,

            # MLP层：保持L4的比例
            "MLP_layer.FC_layers.2.*": 0.86,
            "MLP_layer.FC_layers.1.*": 0.94,
            "MLP_layer.FC_layers.0.*": 0.965,

            "embedding_h.bias": 0.97,
            "embedding_h.weight": 0.998,
            "*.batchnorm_h.*": 0.99,
        }
    },
    9: {
        'description': 'r99 GNN微调组3 (GNN=0.999，接近完全保留)',
        'ratios': {
            # 目的：测试GNN层保留率接近完全保留时的效果
            # 期望：相比0.995可能进一步提升ASR，但收益可能递减
            "layers.3.*": 0.999,
            "layers.2.*": 0.999,
            "layers.1.*": 0.999,
            "layers.0.*": 0.999,

            # MLP层：保持L4的比例
            "MLP_layer.FC_layers.2.*": 0.86,
            "MLP_layer.FC_layers.1.*": 0.94,
            "MLP_layer.FC_layers.0.*": 0.965,

            "embedding_h.bias": 0.97,
            "embedding_h.weight": 0.998,
            "*.batchnorm_h.*": 0.99,
        }
    },
    10: {
        'description': 'r99 GNN微调组4 (GNN=1.0，完全保留)',
        'ratios': {
            # 目的：测试GNN层完全保留时的效果，探索GNN层保留率的上限
            # 期望：达到ASR持久性的理论上限，但可能边际收益很小
            "layers.3.*": 1.0,
            "layers.2.*": 1.0,
            "layers.1.*": 1.0,
            "layers.0.*": 1.0,

            # MLP层：保持L4的比例
            "MLP_layer.FC_layers.2.*": 0.86,
            "MLP_layer.FC_layers.1.*": 0.94,
            "MLP_layer.FC_layers.0.*": 0.965,

            "embedding_h.bias": 0.97,
            "embedding_h.weight": 0.998,
            "*.batchnorm_h.*": 0.99,
        }
    },
    11: {
        'description': 'r99 BatchNorm微调组1 (BatchNorm=0.97)',
        'ratios': {
            # 目的：测试降低BatchNorm层保留率是否能进一步减少良性梯度
            # 期望：可能降低ASR衰减率，提升后门持久性
            "layers.3.*": 0.99,
            "layers.2.*": 0.99,
            "layers.1.*": 0.99,
            "layers.0.*": 0.99,

            # MLP层：保持L4的比例
            "MLP_layer.FC_layers.2.*": 0.86,
            "MLP_layer.FC_layers.1.*": 0.94,
            "MLP_layer.FC_layers.0.*": 0.965,

            "embedding_h.bias": 0.97,
            "embedding_h.weight": 0.998,
            "*.batchnorm_h.*": 0.97,  # 从0.99降到0.97
        }
    },
    12: {
        'description': 'r99 BatchNorm微调组2 (BatchNorm=0.98)',
        'ratios': {
            # 目的：测试BatchNorm层保留率介于0.97和0.99之间时的效果
            # 期望：找到BatchNorm保留率的最佳平衡点
            "layers.3.*": 0.99,
            "layers.2.*": 0.99,
            "layers.1.*": 0.99,
            "layers.0.*": 0.99,

            # MLP层：保持L4的比例
            "MLP_layer.FC_layers.2.*": 0.86,
            "MLP_layer.FC_layers.1.*": 0.94,
            "MLP_layer.FC_layers.0.*": 0.965,

            "embedding_h.bias": 0.97,
            "embedding_h.weight": 0.998,
            "*.batchnorm_h.*": 0.98,  # 介于0.97和0.99之间
        }
    },
    13: {
        'description': 'r99 BatchNorm微调组3 (BatchNorm=1.0，完全保留)',
        'ratios': {
            # 目的：测试BatchNorm层完全保留时是否能提升性能
            # 期望：如果BatchNorm层梯度较小，完全保留可能有利
            "layers.3.*": 0.99,
            "layers.2.*": 0.99,
            "layers.1.*": 0.99,
            "layers.0.*": 0.99,

            # MLP层：保持L4的比例
            "MLP_layer.FC_layers.2.*": 0.86,
            "MLP_layer.FC_layers.1.*": 0.94,
            "MLP_layer.FC_layers.0.*": 0.965,

            "embedding_h.bias": 0.97,
            "embedding_h.weight": 0.998,
            "*.batchnorm_h.*": 1.0,  # 完全保留BatchNorm层
        }
    },
    14: {
        'description': 'r99 BatchNorm微调组4 (BatchNorm=0.995，稍微提高)',
        'ratios': {
            # 目的：测试BatchNorm层较多保留时是否能提升性能
            # 期望：如果BatchNorm层梯度较小，可能有利
            "layers.3.*": 0.99,
            "layers.2.*": 0.99,
            "layers.1.*": 0.99,
            "layers.0.*": 0.99,

            # MLP层：保持L4的比例
            "MLP_layer.FC_layers.2.*": 0.86,
            "MLP_layer.FC_layers.1.*": 0.94,
            "MLP_layer.FC_layers.0.*": 0.965,

            "embedding_h.bias": 0.97,
            "embedding_h.weight": 0.998,
            "*.batchnorm_h.*": 0.995,  # 完全保留BatchNorm层
        }
    },
}


def get_layer_ratios(preset_id):
    """
    Get layer ratios by preset ID.

    Args:
        preset_id: Preset ID (1, 2, 3, ...)

    Returns:
        Dict of layer ratios, or None if preset_id is invalid
    """
    if preset_id in LAYER_PRESETS:
        return LAYER_PRESETS[preset_id]['ratios']
    return None


def get_preset_description(preset_id):
    """
    Get preset description by ID.

    Args:
        preset_id: Preset ID

    Returns:
        Description string, or 'Unknown' if preset_id is invalid
    """
    if preset_id in LAYER_PRESETS:
        return LAYER_PRESETS[preset_id]['description']
    return 'Unknown'


def compute_grad_mask(model, benign_data_loader, loss_func, device, ratio=0.95, aggregate_all_layer=False,
                      layer_ratios=None, preset_id=None, model_name='GCN', dataset='NCI1'):
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
        preset_id: Preset ID to load layer_ratios from LAYER_PRESETS. Takes precedence over layer_ratios.
        model_name: Model name for debugging
        dataset: Dataset name for debugging

    Returns:
        mask_grad_list: List of mask tensors for each parameter
    """
    model.train()
    model.zero_grad()

    # Load layer ratios from preset if specified
    if preset_id is not None:
        preset_ratios = get_layer_ratios(preset_id)
        if preset_ratios is not None:
            layer_ratios = preset_ratios
            # print(f"[INFO] Using layer preset ID {preset_id}: {get_preset_description(preset_id)}")

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
                # print(f"  {name}: ratio={layer_ratio:.4f}, kept={kept_pct:.2f}% ({int(mask.sum().item())}/{mask.numel()})")

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
