from Common.Node.workerbasev2 import WorkerBaseV2
import torch
from torch import nn
from torch import device
import json
import os
from Common.Utils.options import args_parser
from Common.Utils.gnn_util import split_dataset
import time
from Common.Utils.evaluate import gnn_evaluate_accuracy_v2
import numpy as np 
import torch.nn.functional as F
from GNN_common.data.TUs import TUsDataset
from GNN_common.nets.TUs_graph_classification.load_net import gnn_model  # import GNNs
from torch.utils.data import DataLoader
import copy

# 梯度分析相关函数
def split_params_by_layer(model):
    """
    将模型参数按层类型分为三个子集：
    1. 节点特征层：embedding + GAT层
    2. 边权重层：标准GATConv中边权重是动态计算的，无专门参数，此组为空
    3. 全局池化层：MLP读出层
    """
    node_params = []
    edge_params = []
    pool_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'embedding_h' in name:
            # 节点特征嵌入层
            node_params.append((name, param))
        elif 'MLP_layer' in name:
            # 全局池化/读出层
            pool_params.append((name, param))
        elif 'layers' in name and 'gatconv' in name:
            # GAT消息传递层（节点特征层的一部分）
            node_params.append((name, param))
        elif 'layers' in name and 'batchnorm' in name:
            # 批归一化层（节点特征层的一部分）
            node_params.append((name, param))
        # 标准GATConv中边权重是动态计算的，没有专门的边权重参数

    return node_params, edge_params, pool_params


def compute_topk_contribution(grad_list, k_ratios=[0.01, 0.05, 0.10, 0.20, 0.50]):
    """
    计算前k%梯度的平方和占总平方和的比例

    Args:
        grad_list: 参数梯度列表 [(name, grad_tensor), ...]
        k_ratios: k的比例列表，如[0.01, 0.05, 0.10, 0.20, 0.50]

    Returns:
        dict: {k_ratio: contribution_ratio, ...}
    """
    # 将所有梯度展平
    all_grads = []
    for _, grad in grad_list:
        if grad is not None:
            all_grads.append(grad.abs().view(-1).cpu().detach().numpy())

    if not all_grads:
        return {k: 0.0 for k in k_ratios}

    all_grads = np.concatenate(all_grads)
    total_sq_sum = np.sum(all_grads ** 2)

    # 避免除以0
    if total_sq_sum < 1e-10:
        return {k: 0.0 for k in k_ratios}

    # 按绝对值降序排序
    sorted_grads = np.sort(all_grads)[::-1]

    results = {}
    for k in k_ratios:
        k_count = int(len(sorted_grads) * k)
        if k_count == 0:
            results[k] = 0.0
        else:
            topk_sq_sum = np.sum(sorted_grads[:k_count] ** 2)
            results[k] = topk_sq_sum / total_sq_sum

    return results


def analyze_gradients(client, epoch, output_file='gradient_analysis.txt'):
    """
    分析客户端的梯度分布

    Args:
        client: 客户端对象
        epoch: 当前轮次
        output_file: 输出文件路径
    """
    model = client.model
    device = client.device

    # 重新计算梯度以确保梯度存在（避免被optimizer.zero_grad()清零）
    model.train()
    model.zero_grad()

    # 使用训练数据的一个小批次来计算梯度
    try:
        batch_graphs, batch_labels = next(iter(client.train_iter))
    except StopIteration:
        # 如果迭代器已耗尽，重新创建
        batch_graphs, batch_labels = next(iter(client.train_iter))

    batch_graphs = batch_graphs.to(device)
    batch_x = batch_graphs.ndata['feat'].to(device)
    batch_e = batch_graphs.edata['feat'].to(device)
    batch_labels = batch_labels.to(torch.long).to(device)

    batch_scores = model.forward(batch_graphs, batch_x, batch_e)
    loss = model.loss(batch_scores, batch_labels)
    loss.backward()

    # 按层分组参数
    node_params, edge_params, pool_params = split_params_by_layer(model)

    # 收集各层的梯度
    node_grad_list = [(name, param.grad) for name, param in node_params if param.grad is not None]
    edge_grad_list = [(name, param.grad) for name, param in edge_params if param.grad is not None]
    pool_grad_list = [(name, param.grad) for name, param in pool_params if param.grad is not None]
    all_grad_list = node_grad_list + edge_grad_list + pool_grad_list

    # 计算各层的贡献度
    node_results = compute_topk_contribution(node_grad_list)
    edge_results = compute_topk_contribution(edge_grad_list)
    pool_results = compute_topk_contribution(pool_grad_list)
    all_results = compute_topk_contribution(all_grad_list)

    # ========== 新增：跨层分析 ==========
    # 将所有梯度展平并记录来源层
    grad_array = []
    layer_source = []  # 记录每个梯度元素来自哪一层

    for name, grad in node_grad_list:
        grad_flat = grad.abs().view(-1).cpu().detach().numpy()
        grad_array.append(grad_flat)
        layer_source.extend(['node'] * len(grad_flat))

    for name, grad in pool_grad_list:
        grad_flat = grad.abs().view(-1).cpu().detach().numpy()
        grad_array.append(grad_flat)
        layer_source.extend(['pool'] * len(grad_flat))

    grad_array = np.concatenate(grad_array)
    layer_source = np.array(layer_source)

    # 计算全局Top-k梯度中各层的占比
    total_elements = len(grad_array)
    layer_contribution_in_topk = {}

    for k in [0.01, 0.05, 0.10, 0.20, 0.50]:
        k_count = int(total_elements * k)
        if k_count == 0:
            continue

        # 找到Top-k梯度的索引
        topk_indices = np.argpartition(grad_array, -k_count)[-k_count:]

        # 统计各层在Top-k中的数量
        node_in_topk = np.sum(layer_source[topk_indices] == 'node')
        pool_in_topk = np.sum(layer_source[topk_indices] == 'pool')

        layer_contribution_in_topk[k] = {
            'node_ratio': node_in_topk / k_count if k_count > 0 else 0,
            'pool_ratio': pool_in_topk / k_count if k_count > 0 else 0
        }

    # 计算各层对全局L2范数的贡献
    total_l2 = np.sum(grad_array ** 2)
    # node_l2 = np.sum([g.abs().cpu().detach().numpy() ** 2 for _, g in node_grad_list])
    # pool_l2 = np.sum([g.abs().cpu().detach().numpy() ** 2 for _, g in pool_grad_list])
    node_l2 = np.sum([g.pow(2).sum().item() for _, g in node_grad_list])
    pool_l2 = np.sum([g.pow(2).sum().item() for _, g in pool_grad_list])

    node_l2_ratio = node_l2 / total_l2 if total_l2 > 0 else 0
    pool_l2_ratio = pool_l2 / total_l2 if total_l2 > 0 else 0

    # 输出到文件
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"Epoch {epoch}\n")
        f.write("=" * 80 + "\n")

        # 节点特征层
        f.write(f"[节点特征层] 参数数量: {len(node_params)}\n")
        for k in [0.01, 0.05, 0.10, 0.20, 0.50]:
            f.write(f"  前{k*100:.0f}%梯度贡献度: {node_results[k]:.4f}\n")

        # 边权重层（标准GATConv中无此层参数）
        f.write(f"[边权重层] 参数数量: {len(edge_params)}\n")
        f.write(f"  (标准GATConv中边权重是动态计算的，无专门参数)\n")

        # 全局池化层
        f.write(f"[全局池化层] 参数数量: {len(pool_params)}\n")
        for k in [0.01, 0.05, 0.10, 0.20, 0.50]:
            f.write(f"  前{k*100:.0f}%梯度贡献度: {pool_results[k]:.4f}\n")

        # 全部参数
        f.write(f"[全部参数] 参数数量: {len(node_params) + len(edge_params) + len(pool_params)}\n")
        for k in [0.01, 0.05, 0.10, 0.20, 0.50]:
            f.write(f"  前{k*100:.0f}%梯度贡献度: {all_results[k]:.4f}\n")

        # ========== 新增：跨层分析输出 ==========
        f.write("\n" + "-" * 80 + "\n")
        f.write("[跨层分析]\n")

        f.write("各层对全局L2范数的贡献:\n")
        f.write(f"  节点特征层: {node_l2_ratio:.2%}\n")
        f.write(f"  全局池化层: {pool_l2_ratio:.2%}\n")

        f.write("\n全局Top-k梯度中各层的占比:\n")
        f.write(f"{'Top-k':<10} {'节点特征层':<15} {'全局池化层':<15}\n")
        f.write("-" * 50)
        for k in [0.01, 0.05, 0.10, 0.20, 0.50]:
            node_ratio = layer_contribution_in_topk[k]['node_ratio']
            pool_ratio = layer_contribution_in_topk[k]['pool_ratio']
            f.write(f"\n前{k*100:6.0f}%  {node_ratio:>6.2%}         {pool_ratio:>6.2%}")

        f.write("\n\n")

    return {
        'node': node_results,
        'edge': edge_results,
        'pool': pool_results,
        'all': all_results,
        'node_l2_ratio': node_l2_ratio,
        'pool_l2_ratio': pool_l2_ratio,
        'layer_contribution_in_topk': layer_contribution_in_topk
    }

def server_robust_agg(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

class ClearDenseClient(WorkerBaseV2):
    def __init__(self, client_id, model, loss_func, train_iter, attack_iter, test_iter, config, optimizer, device, grad_stub, args, scheduler):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter, attack_iter=attack_iter, test_iter=test_iter, config=config, optimizer=optimizer, device=device)
        self.client_id = client_id
        self.grad_stub = None
        self.args = args
        self.scheduler = scheduler

    def update(self):
        pass

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)
    with open(args.config) as f:
        config = json.load(f)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = TUsDataset(args)

    collate = dataset.collate
    MODEL_NAME = config['model']
    net_params = config['net_params']
    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    net_params['in_dim'] = dataset.all.graph_lists[0].ndata['feat'][0].shape[0]
    num_classes = torch.max(dataset.all.graph_labels).item() + 1
    net_params['n_classes'] = num_classes
    net_params['dropout'] = args.dropout

    global_model = gnn_model(MODEL_NAME, net_params)
    global_model = global_model.to(device)
    #print("Target Model:\n{}".format(model))
    client = []
    loss_func = nn.CrossEntropyLoss()
    # Load data
    partition, avg_nodes = split_dataset(args, dataset)
    drop_last = True if MODEL_NAME == 'DiffPool' else False
    triggers = []
    for i in range(args.num_workers):
        local_model = copy.deepcopy(global_model)
        local_model = local_model.to(device)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
        train_dataset = partition[i]
        test_dataset = partition[-1]
        print("Client %d training data num: %d"%(i, len(train_dataset)))
        print("Client %d testing data num: %d"%(i, len(test_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                     drop_last=drop_last,
                                     collate_fn=dataset.collate)
        attack_loader = None
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                     drop_last=drop_last,
                                     collate_fn=dataset.collate)
        
        client.append(ClearDenseClient(client_id=i, model=local_model, loss_func=loss_func, train_iter=train_loader, attack_iter=attack_loader, test_iter=test_loader, config=config, optimizer=optimizer, device=device, grad_stub=None, args=args, scheduler=scheduler))
    
    # check model memory address
    for i in range(args.num_workers):
        add_m = id(client[i].model)
        add_o = id(client[i].optimizer)
        print('model {} address: {}'.format(i, add_m))
        print('optimizer {} address: {}'.format(i, add_o))

    acc_record = [0]
    counts = 0
    for epoch in range(args.epochs):
        print('epoch:',epoch)

        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for i in range(args.num_workers):
            att_list = []
            train_loss, train_acc, test_loss, test_acc = client[i].gnn_train_v2()
            client[i].scheduler.step()
            print('Client %d, loss %.4f, train acc %.3f, test loss %.4f, test acc %.3f'
                    % (i, train_loss, train_acc, test_loss, test_acc))
            if not args.filename == "":
                save_path = os.path.join(args.filename, str(args.seed), config['model'] + '_' + args.dataset + \
                    '_%d_%d_%.2f_%.2f_%.2f'%(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_%d.txt'%i)
                path = os.path.split(save_path)[0]
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)

                with open(save_path, 'a') as f:
                    f.write('%.3f %.3f %.3f %.3f'%(train_loss, train_acc, test_loss, test_acc))
                    f.write('\n')

        weights = []
        # 每隔10轮，对client[0]的梯度进行分布分析
        if epoch % 10 == 0:
            grad_analysis_file = os.path.join(args.filename, 'gradient_analysis.txt') if args.filename else 'gradient_analysis.txt'
            # 确保输出目录存在
            grad_dir = os.path.dirname(grad_analysis_file)
            if grad_dir and not os.path.exists(grad_dir):
                os.makedirs(grad_dir, exist_ok=True)
            analyze_gradients(client[0], epoch, grad_analysis_file)
            print(f'[梯度分析] Epoch {epoch} 的梯度分布已保存至 {grad_analysis_file}')

        for i in range(args.num_workers):
            weights.append(client[i].get_weights())
        # Aggregation in the server to get the global model
        result = server_robust_agg(weights)

        for i in range(args.num_workers):
            client[i].set_weights(weights=result)
            client[i].upgrade()
        # update global model's weights
        global_model.load_state_dict(result)

        # evaluate the global model: test_acc
        test_acc = gnn_evaluate_accuracy_v2(client[0].test_iter, global_model)
        print('Global Test Acc: %.3f'%test_acc)
        if not args.filename == "":
            save_path = os.path.join(args.filename, str(args.seed), MODEL_NAME + '_' + args.dataset + '_%d_%d_%.2f_%.2f_%.2f'\
                        %(args.num_workers, args.num_mali, args.frac_of_avg, args.poisoning_intensity, args.density) + '_global_test.txt')
            path = os.path.split(save_path)[0]
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)

            with open(save_path, 'a') as f:
                f.write("%.3f" % (test_acc))
                f.write("\n")

