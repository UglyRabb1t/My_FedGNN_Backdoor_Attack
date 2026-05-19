"""
Warmup Training Script for GradMask Analysis
=============================================
此脚本进行预热阶段训练（仅使用良性数据），训练指定轮数后保存全局模型。

用途：
1. 为梯度分析提供预热后的模型
2. 预热后的模型梯度分布更符合实际 GradMask 使用场景
"""

import argparse
import torch
from torch import nn
import json
import os
import time
import copy

from Common.Utils.options import args_parser
from Common.Utils.gnn_util import load_pkl, split_dataset
from Common.Utils.evaluate import gnn_evaluate_accuracy_v2
from GNN_common.data.TUs import TUsDataset
from GNN_common.nets.TUs_graph_classification.load_net import gnn_model
from torch.utils.data import DataLoader
from GNN_common.train.metrics import accuracy_TU as accuracy


def server_robust_agg(w):
    """服务器聚合函数"""
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


class WarmupClient:
    """预热阶段客户端，只进行良性训练"""

    def __init__(self, client_id, model, loss_func, train_iter, test_iter,
                 optimizer, device, scheduler):
        self.client_id = client_id
        self.model = model
        self.loss_func = loss_func
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler

    def gnn_train_v2(self):
        """本地训练"""
        self.model.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()

        for batch_graphs, batch_labels in self.train_iter:
            batch_graphs = batch_graphs.to(self.device)
            batch_x = batch_graphs.ndata['feat'].to(self.device)
            batch_e = batch_graphs.edata['feat'].to(self.device)
            batch_labels = batch_labels.to(torch.long).to(self.device)

            batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
            l = self.model.loss(batch_scores, batch_labels)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()

            train_l_sum += l.cpu().item()
            train_acc_sum += accuracy(batch_scores, batch_labels)
            n += batch_labels.size(0)
            batch_count += 1

        self._weights = self.model.state_dict()

        # 评估
        test_acc, test_l = self.gnn_evaluate()

        return train_l_sum / batch_count, train_acc_sum / n, test_l, test_acc

    def gnn_evaluate(self):
        """评估模型"""
        self.model.eval()
        test_l_sum, test_acc_sum, n, batch_count = 0.0, 0.0, 0, 0

        with torch.no_grad():
            for batch_graphs, batch_labels in self.test_iter:
                batch_graphs = batch_graphs.to(self.device)
                batch_x = batch_graphs.ndata['feat'].to(self.device)
                batch_e = batch_graphs.edata['feat'].to(self.device)
                batch_labels = batch_labels.to(torch.long).to(self.device)

                batch_scores = self.model.forward(batch_graphs, batch_x, batch_e)
                l = self.model.loss(batch_scores, batch_labels)

                test_l_sum += l.cpu().item()
                test_acc_sum += accuracy(batch_scores, batch_labels)
                n += batch_labels.size(0)
                batch_count += 1

        return test_acc_sum / n, test_l_sum / batch_count

    def get_weights(self):
        """获取模型权重"""
        return self._weights

    def set_weights(self, weights):
        """设置模型权重"""
        self.model.load_state_dict(weights)

    def upgrade(self):
        """升级模型（占位函数）"""
        pass


if __name__ == '__main__':
    # 先提取自定义参数，再调用 args_parser()
    import sys

    warmup_epochs = 30
    save_model = None

    if '--warmup_epochs' in sys.argv:
        idx = sys.argv.index('--warmup_epochs')
        warmup_epochs = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 30
        # 从 sys.argv 中移除这两个参数，避免 args_parser() 报错
        sys.argv.pop(idx + 1)
        sys.argv.pop(idx)

    if '--save_model' in sys.argv:
        idx = sys.argv.index('--save_model')
        save_model = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        sys.argv.pop(idx + 1)
        sys.argv.pop(idx)

    args = args_parser()

    # 设置自定义参数
    args.warmup_epochs = warmup_epochs
    args.save_model = save_model

    # 加载配置
    with open(args.config) as f:
        config = json.load(f)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据集
    dataset = TUsDataset(args)
    MODEL_NAME = config['model']
    net_params = config['net_params']

    # 为 GCN/GAT 添加自环
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] 为 GCN/GAT 模型添加图自环。")

    # 设置网络参数
    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    net_params['n_classes'] = num_classes
    net_params['dropout'] = args.dropout

    # 创建全局模型
    global_model = gnn_model(MODEL_NAME, net_params)
    global_model = global_model.to(device)

    print(f"模型: {MODEL_NAME}")
    print(f"数据集: {config['dataset']}")
    print(f"预热轮数: {args.warmup_epochs}")

    # 加载和分割数据
    partition, avg_nodes = split_dataset(args, dataset)

    # 初始化客户端
    loss_func = nn.CrossEntropyLoss()
    clients = []

    drop_last = MODEL_NAME == 'DiffPool'

    for i in range(args.num_workers):
        local_model = copy.deepcopy(global_model)
        local_model = local_model.to(device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)

        train_loader = DataLoader(partition[i], batch_size=args.batch_size, shuffle=True,
                                drop_last=drop_last, collate_fn=dataset.collate)
        test_loader = DataLoader(partition[-1], batch_size=args.batch_size, shuffle=True,
                               drop_last=drop_last, collate_fn=dataset.collate)

        client = WarmupClient(client_id=i, model=local_model, loss_func=loss_func,
                             train_iter=train_loader, test_iter=test_loader,
                             optimizer=optimizer, device=device, scheduler=scheduler)
        clients.append(client)

        print(f"客户端 {i}: 训练数据 {len(partition[i])}, 测试数据 {len(partition[-1])}")

    print(f"\n{'='*80}")
    print(f"开始预热训练（仅良性数据）")
    print(f"{'='*80}\n")

    # 预热训练
    for epoch in range(args.warmup_epochs):
        print(f'预热轮次: {epoch + 1}/{args.warmup_epochs}')

        # 客户端本地训练
        for i in range(args.num_workers):
            train_loss, train_acc, test_loss, test_acc = clients[i].gnn_train_v2()
            clients[i].scheduler.step()

            print(f'客户端 {i}: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.3f}, '
                  f'测试损失={test_loss:.4f}, 测试准确率={test_acc:.3f}')

        # 服务器聚合
        weights = [client.get_weights() for client in clients]
        result = server_robust_agg(weights)

        # 分发新权重到客户端
        for client in clients:
            client.set_weights(weights=result)
            client.upgrade()

        # 更新全局模型
        global_model.load_state_dict(result)

        # 评估全局模型
        test_acc = gnn_evaluate_accuracy_v2(clients[0].test_iter, global_model)
        print(f'全局模型测试准确率: {test_acc:.3f}\n')

    print(f"{'='*80}")
    print(f"预热训练完成！")
    print(f"{'='*80}")

    # 保存预热后的全局模型
    if args.save_model:
        os.makedirs(args.save_model, exist_ok=True)
        model_path = os.path.join(args.save_model, f'warmup_{MODEL_NAME}_{config["dataset"]}_seed{args.seed}_ep{args.warmup_epochs}.pth')

        torch.save({
            'model_state_dict': global_model.state_dict(),
            'net_params': net_params,
            'model_name': MODEL_NAME,
            'dataset': config['dataset'],
            'seed': args.seed,
            'warmup_epochs': args.warmup_epochs,
            'config': config
        }, model_path)

        print(f"\n预热后的模型已保存到: {model_path}")
        print(f"\n现在可以使用以下命令进行梯度分析:")
        print(f"  python gradient_analyzer.py --config {args.config} --dataset {config['dataset']} "
              f"--model {MODEL_NAME} --seed {args.seed} --warmup_epochs {args.warmup_epochs} "
              f"--load_model {model_path} --filename ./analyzer_results/")

        # 同时保存一个简化版本的模型文件（只包含state_dict）
        simple_model_path = os.path.join(args.save_model, f'warmup_{MODEL_NAME}_{config["dataset"]}_seed{args.seed}_ep{args.warmup_epochs}_simple.pth')
        torch.save(global_model.state_dict(), simple_model_path)
        print(f"\n简化模型文件（仅state_dict）: {simple_model_path}")
