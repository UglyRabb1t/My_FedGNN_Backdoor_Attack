"""
验证 pyhessian 是否可用于 GNN 模型
"""
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import dgl
import numpy as np
from pyhessian import hessian

# 导入GNN模型
from GNN_common.nets.TUs_graph_classification.load_net import gnn_model


class GNNWrapper(nn.Module):
    """
    包装GNN模型，使其forward接收单个输入张量
    pyhessian期望的格式是 model(inputs) -> outputs
    """
    def __init__(self, gnn_model, g, h, e):
        super().__init__()
        self.gnn_model = gnn_model
        self.g = g  # 固定的图结构
        self.h = h  # 固定的节点特征
        self.e = e  # 固定的边特征

    def forward(self, _):
        """
        忽略输入参数，使用预先存储的图数据进行前向传播
        pyhessian会调用 model(inputs)，我们忽略inputs返回GNN输出
        """
        return self.gnn_model(self.g, self.h, self.e)


def create_simple_test_data():
    """创建简单的测试数据"""
    # 创建3个节点，4条边的图
    num_nodes = 3
    node_feat_dim = 5
    edge_feat_dim = 3
    num_classes = 2

    g = dgl.graph(([0, 1, 2, 0], [1, 2, 0, 2]))
    h = torch.randn(num_nodes, node_feat_dim)
    e = torch.randn(g.num_edges(), edge_feat_dim)
    target = torch.tensor([0], dtype=torch.long)

    return g, h, e, target, num_classes, node_feat_dim, edge_feat_dim


def test_pyhessian_basic():
    """测试pyhessian基本功能"""
    print("=" * 60)
    print("测试1: pyhessian 基本功能（简单MLP）")
    print("=" * 60)

    # 创建一个简单的MLP来测试pyhessian本身是否工作
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = SimpleMLP()
    inputs = torch.randn(4, 10)
    targets = torch.randint(0, 2, (4,))

    criterion = nn.CrossEntropyLoss()

    try:
        hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=False)
        trace = hessian_comp.trace()
        print(f"成功计算Hessian trace: {np.mean(trace):.4f}")
        return True
    except Exception as e:
        print(f"pyhessian基本测试失败: {e}")
        return False


def test_pyhessian_gnn_with_wrapper():
    """测试使用wrapper的GNN模型"""
    print("\n" + "=" * 60)
    print("测试2: pyhessian + GNN Wrapper")
    print("=" * 60)

    # 创建测试数据
    g, h, e, target, num_classes, node_feat_dim, edge_feat_dim = create_simple_test_data()

    # 创建GNN模型配置
    net_params = {
        'in_dim': node_feat_dim,
        'hidden_dim': 8,
        'out_dim': 8,
        'n_classes': num_classes,
        'in_feat_dropout': 0.0,
        'dropout': 0.0,
        'L': 2,
        'readout': 'mean',
        'batch_norm': False,
        'residual': False
    }

    # 创建GNN模型
    gnn = gnn_model('GCN', net_params)
    gnn.eval()

    # 包装GNN模型
    dummy_input = torch.tensor([0.0])  # pyhessian需要一个输入
    wrapped_model = GNNWrapper(gnn, g, h, e)

    criterion = nn.CrossEntropyLoss()

    try:
        # 测试wrapper是否工作
        output = wrapped_model(dummy_input)
        print(f"Wrapper前向传播测试成功，输出shape: {output.shape}")

        # 尝试使用pyhessian
        hessian_comp = hessian(wrapped_model, criterion, data=(dummy_input, target), cuda=False)
        trace = hessian_comp.trace()
        print(f"成功计算GNN的Hessian trace: {np.mean(trace):.4f}")
        return True

    except Exception as e:
        print(f"pyhessian + GNN Wrapper测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pyhessian_batch_gnn():
    """测试批量图的Hessian计算"""
    print("\n" + "=" * 60)
    print("测试3: 批量图数据的Hessian计算")
    print("=" * 60)

    # 创建批量图
    g1 = dgl.graph(([0, 1], [1, 0]))
    g1.ndata['feat'] = torch.randn(2, 5)
    g1.edata['feat'] = torch.randn(2, 3)

    g2 = dgl.graph(([0], [0]))
    g2.ndata['feat'] = torch.randn(1, 5)
    g2.edata['feat'] = torch.randn(1, 3)

    batch_g = dgl.batch([g1, g2])
    batch_h = batch_g.ndata['feat']
    batch_e = batch_g.edata['feat']
    batch_labels = torch.tensor([0, 1], dtype=torch.long)

    # 创建GNN模型
    net_params = {
        'in_dim': 5,
        'hidden_dim': 8,
        'out_dim': 8,
        'n_classes': 2,
        'in_feat_dropout': 0.0,
        'dropout': 0.0,
        'L': 2,
        'readout': 'mean',
        'batch_norm': False,
        'residual': False
    }

    gnn = gnn_model('GCN', net_params)
    gnn.eval()

    class BatchGNNWrapper(nn.Module):
        def __init__(self, gnn_model, g, h, e):
            super().__init__()
            self.gnn_model = gnn_model
            self.g = g
            self.h = h
            self.e = e

        def forward(self, _):
            return self.gnn_model(self.g, self.h, self.e)

    dummy_input = torch.tensor([0.0])
    wrapped_model = BatchGNNWrapper(gnn, batch_g, batch_h, batch_e)
    criterion = nn.CrossEntropyLoss()

    try:
        output = wrapped_model(dummy_input)
        print(f"批量图Wrapper前向传播成功，输出shape: {output.shape}")

        hessian_comp = hessian(wrapped_model, criterion, data=(dummy_input, batch_labels), cuda=False)
        trace = hessian_comp.trace()
        print(f"成功计算批量图的Hessian trace: {np.mean(trace):.4f}")
        return True

    except Exception as e:
        print(f"批量图Hessian计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pyhessian_eigenvalues():
    """测试计算Hessian特征值"""
    print("\n" + "=" * 60)
    print("测试4: Hessian特征值计算")
    print("=" * 60)

    g, h, e, target, num_classes, node_feat_dim, edge_feat_dim = create_simple_test_data()

    net_params = {
        'in_dim': node_feat_dim,
        'hidden_dim': 8,
        'out_dim': 8,
        'n_classes': num_classes,
        'in_feat_dropout': 0.0,
        'dropout': 0.0,
        'L': 2,
        'readout': 'mean',
        'batch_norm': False,
        'residual': False
    }

    gnn = gnn_model('GCN', net_params)
    gnn.eval()

    dummy_input = torch.tensor([0.0])
    wrapped_model = GNNWrapper(gnn, g, h, e)
    criterion = nn.CrossEntropyLoss()

    try:
        hessian_comp = hessian(wrapped_model, criterion, data=(dummy_input, target), cuda=False)
        top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
        print(f"成功计算Hessian特征值: {top_eigenvalues}")
        return True

    except Exception as e:
        print(f"Hessian特征值计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("开始验证 pyhessian 与 GNN 的兼容性...\n")

    results = []

    # 运行测试
    results.append(("基本功能测试", test_pyhessian_basic()))
    results.append(("GNN Wrapper测试", test_pyhessian_gnn_with_wrapper()))
    results.append(("批量图测试", test_pyhessian_batch_gnn()))
    results.append(("特征值测试", test_pyhessian_eigenvalues()))

    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, success in results:
        status = "通过" if success else "失败"
        print(f"{name:20s}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n所有测试通过！pyhessian可以用于GNN模型。")
        print("\n集成建议：")
        print("1. 使用GNNWrapper包装模型")
        print("2. 将图数据固定在wrapper中")
        print("3. 调用hessian()时传入dummy输入和真实标签")
    else:
        print("\n部分测试失败，需要进一步调试。")
