"""
验证 pyhessian 是否可用于 GNN 模型 + 实现简化版 Hessian Trace
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import dgl
import numpy as np
from pyhessian import hessian

# 导入GNN模型
from GNN_common.nets.TUs_graph_classification.load_net import gnn_model


class GNNWrapper(nn.Module):
    """包装GNN模型"""
    def __init__(self, gnn_model, g, h, e):
        super().__init__()
        self.gnn_model = gnn_model
        self.g = g
        self.h = h
        self.e = e

    def forward(self, _):
        return self.gnn_model(self.g, self.h, self.e)


def diagnose_gradient_issue():
    """诊断GNN模型梯度问题"""
    print("=" * 60)
    print("诊断: GNN模型梯度问题")
    print("=" * 60)

    # 创建简单数据
    g = dgl.graph(([0, 1], [1, 0]))
    h = torch.randn(2, 5)
    e = torch.randn(2, 3)
    target = torch.tensor([0], dtype=torch.long)

    net_params = {
        'in_dim': 5, 'hidden_dim': 8, 'out_dim': 8, 'n_classes': 2,
        'in_feat_dropout': 0.0, 'dropout': 0.0, 'L': 2,
        'readout': 'mean', 'batch_norm': False, 'residual': False
    }

    gnn = gnn_model('GCN', net_params)
    criterion = nn.CrossEntropyLoss()

    print("\n第一次前向传播...")
    output = gnn(g, h, e)
    loss = criterion(output, target)

    print("第一次反向传播...")
    loss.backward()

    print("\n检查模型参数梯度:")
    for name, param in gnn.named_parameters():
        if param.requires_grad:
            print(f"  {name}: shape={param.shape}, grad={type(param.grad)}, has_grad={param.grad is not None}")
            if param.grad is not None:
                print(f"         grad shape={param.grad.shape}")

    # 尝试计算HVP
    print("\n尝试手动计算Hessian-vector-product...")

    # 获取参数和梯度 - 只保留有梯度的参数
    params = []
    grads = []
    for p in gnn.parameters():
        if p.requires_grad and p.grad is not None:
            params.append(p)
            grads.append(p.grad)

    print(f"有梯度的参数数量: {len(params)} / {len(list(gnn.parameters()))}")

    # 创建随机向量v
    v = [torch.randn_like(g) for g in grads]

    try:
        hv = torch.autograd.grad(grads, params, grad_outputs=v, only_inputs=True, retain_graph=True)
        print("成功计算HVP！")
        print(f"  HVP包含{len(hv)}个元素")
        for i, (h, g, p) in enumerate(zip(hv, grads, params)):
            if h is not None:
                print(f"  [{i}] {type(h)}, shape={h.shape}, param={list(p.shape)}")
            else:
                print(f"  [{i}] None - 参数: {list(p.shape)}")
    except Exception as e:
        print(f"HVP计算失败: {e}")
        import traceback
        traceback.print_exc()


def simplified_hessian_trace_gnn(model, g, h, e, labels, criterion, num_samples=10):
    """
    使用 Hutchinson's estimator 简化计算 Hessian trace
    这是一个不依赖 pyhessian 的实现，专门用于 GNN
    """
    print("\n" + "=" * 60)
    print("测试: 简化版 Hessian Trace (Hutchinson's estimator)")
    print("=" * 60)

    model.eval()
    trace_estimate = 0.0

    # 前向传播
    output = model(g, h, e)
    loss = criterion(output, labels)

    # 第一次反向传播
    loss.backward(create_graph=True)  # 需要计算图

    # 获取有梯度的参数
    params = []
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            params.append(p)
            grads.append(p.grad)

    print(f"  使用 {len(params)} 个参数计算trace")

    # Hutchinson's estimator: trace(H) ≈ E[v^T H v] = E[v^T * Hv]
    # 通过随机投影估计
    for _ in range(num_samples):
        # 生成随机向量 v (Rademacher分布: ±1)
        v = [torch.randint(0, 2, g.shape).float() * 2 - 1 for g in grads]

        # 计算 Hv = Hessian * v
        try:
            hv = torch.autograd.grad(grads, params, grad_outputs=v, only_inputs=True, retain_graph=True)
            # 计算 v^T * Hv
            product = sum([torch.sum(v_i * h_i) for v_i, h_i in zip(v, hv) if h_i is not None])
            trace_estimate += product.item()
        except Exception as e:
            print(f"  第{_+1}次采样失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    trace_estimate = trace_estimate / num_samples
    print(f"  样本数: {num_samples}")
    print(f"  Hessian Trace 估计值: {trace_estimate:.4f}")

    return trace_estimate


def simplified_eigenvalue_gnn(model, g, h, e, labels, criterion, top_n=1, max_iter=100):
    """
    使用幂迭代法简化计算 Hessian 的最大特征值
    """
    print("\n" + "=" * 60)
    print("测试: 简化版最大特征值 (幂迭代法)")
    print("=" * 60)

    model.eval()

    # 前向传播
    output = model(g, h, e)
    loss = criterion(output, labels)

    # 获取有梯度的参数
    params = []
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            params.append(p)
            grads.append(p.grad)

    print(f"  使用 {len(params)} 个参数计算特征值")

    # 初始化随机向量
    v = [torch.randn_like(p) for p in params]
    v = [vi / (torch.norm(vi) + 1e-10) for vi in v]

    for i in range(max_iter):
        # 计算 Hessian-vector product
        loss.backward(create_graph=True, retain_graph=True)

        # 更新梯度（只取有梯度的）
        grads = [p.grad for p in params]

        try:
            hv = torch.autograd.grad(grads, params, grad_outputs=v, only_inputs=True, retain_graph=True)
        except Exception as e:
            print(f"  迭代{i+1}失败: {e}")
            import traceback
            traceback.print_exc()
            return None

        # Rayleigh quotient: λ = (v^T Hv) / (v^T v)
        numerator = sum([torch.sum(v_i * h_i) for v_i, h_i in zip(v, hv) if h_i is not None])
        denominator = sum([torch.sum(v_i * v_i) for v_i in v])
        eigenvalue = (numerator / (denominator + 1e-10)).item()

        # 更新 v
        v = [h_i / (torch.norm(h_i) + 1e-10) if h_i is not None else torch.randn_like(v_i)
             for h_i, v_i in zip(hv, v)]

        if (i + 1) % 20 == 0:
            print(f"  迭代 {i+1}: 特征值 ≈ {eigenvalue:.6f}")

    print(f"  收敛后的最大特征值: {eigenvalue:.6f}")
    return eigenvalue


def test_pyhessian_with_troubleshooting():
    """深入测试pyhessian的问题"""
    print("\n" + "=" * 60)
    print("测试: pyhessian 深入调试")
    print("=" * 60)

    g = dgl.graph(([0, 1], [1, 0]))
    h = torch.randn(2, 5)
    e = torch.randn(2, 3)
    target = torch.tensor([0], dtype=torch.long)

    net_params = {
        'in_dim': 5, 'hidden_dim': 8, 'out_dim': 8, 'n_classes': 2,
        'in_feat_dropout': 0.0, 'dropout': 0.0, 'L': 2,
        'readout': 'mean', 'batch_norm': False, 'residual': False
    }

    gnn = gnn_model('GCN', net_params)
    gnn.eval()

    dummy_input = torch.tensor([0.0])
    wrapped_model = GNNWrapper(gnn, g, h, e)
    criterion = nn.CrossEntropyLoss()

    try:
        print("创建hessian对象...")
        hessian_comp = hessian(wrapped_model, criterion, data=(dummy_input, target), cuda=False)
        print("hessian对象创建成功")

        print("\n尝试获取梯度...")
        gradsH = hessian_comp.gradsH
        print(f"gradsH 类型: {type(gradsH)}, 长度: {len(gradsH)}")

        print("\n检查gradsH中每个元素的类型:")
        for i, g in enumerate(gradsH):
            print(f"  [{i}] type={type(g)}, shape={g.shape if hasattr(g, 'shape') else 'N/A'}, is_Tensor={isinstance(g, torch.Tensor)}")

        print("\n尝试计算trace...")
        trace = hessian_comp.trace()
        print(f"成功! trace = {trace}")

    except Exception as e:
        print(f"失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    print("开始验证 pyhessian 与 GNN 的兼容性...\n")

    # 1. 诊断梯度问题
    diagnose_gradient_issue()

    # 2. 测试简化版Hessian trace
    g = dgl.graph(([0, 1], [1, 0]))
    h = torch.randn(2, 5)
    e = torch.randn(2, 3)
    target = torch.tensor([0], dtype=torch.long)
    net_params = {
        'in_dim': 5, 'hidden_dim': 8, 'out_dim': 8, 'n_classes': 2,
        'in_feat_dropout': 0.0, 'dropout': 0.0, 'L': 2,
        'readout': 'mean', 'batch_norm': False, 'residual': False
    }
    gnn = gnn_model('GCN', net_params)
    criterion = nn.CrossEntropyLoss()

    simplified_hessian_trace_gnn(gnn, g, h, e, target, criterion)

    # 3. 测试简化版特征值
    simplified_eigenvalue_gnn(gnn, g, h, e, target, criterion)

    # 4. 深入调试pyhessian
    test_pyhessian_with_troubleshooting()
