"""
为不同数据集和模型生成全局触发器
按照 GAT_NCI1_5_1_0_0.20_0.20_0.80.pkl 的参数设置
"""

import networkx as nx
import pickle
import os
from pathlib import Path
from GNN_common.data.TUs import TUsDataset
from Common.Utils.options import args_parser
import torch
import json
import argparse


def calculate_avg_nodes(dataset):
    """计算数据集的平均节点数"""
    dataset_all = dataset.train[0] + dataset.val[0] + dataset.test[0]

    graph_sizes = []
    for data in dataset_all:
        graph_sizes.append(data[0].num_nodes())
    graph_sizes.sort()

    # 去除异常值（前后30%）
    n = int(0.3 * len(graph_sizes))
    graph_size_normal = graph_sizes[n:len(graph_sizes) - n]

    count = 0
    for size in graph_size_normal:
        count += size
    avg_nodes = count / len(graph_size_normal)
    avg_nodes = round(avg_nodes)

    return avg_nodes


def generate_trigger(avg_nodes, frac_of_avg, density):
    """生成触发器图"""
    num_trigger_nodes = int(avg_nodes * frac_of_avg)
    G_trigger = nx.erdos_renyi_graph(num_trigger_nodes, density, directed=False)
    return G_trigger


def save_trigger(trigger, filepath):
    """保存触发器到文件"""
    savedir = os.path.split(filepath)[0]
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(filepath, 'wb') as output:
        pickle.dump([trigger], output, pickle.HIGHEST_PROTOCOL)
    print(f"触发器已保存到: {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (NCI1, PROTEINS_full, MUTAG等)')
    parser.add_argument('--model', type=str, required=True, help='模型名称 (GCN, GAT)')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--num_workers', type=int, default=5, help='客户端总数')
    parser.add_argument('--num_mali', type=int, default=1, help='恶意客户端数')
    parser.add_argument('--epoch_backdoor', type=int, default=0, help='开始攻击的轮次')
    parser.add_argument('--frac_of_avg', type=float, default=0.20, help='触发器节点占平均节点数的比例')
    parser.add_argument('--poisoning_intensity', type=float, default=0.20, help='投毒强度')
    parser.add_argument('--density', type=float, default=0.80, help='触发器边密度')
    parser.add_argument('--datadir', type=str, default='./Data', help='数据集路径')
    parser.add_argument('--config', type=str, default='./config/GCN_NCI1_config.json', help='配置文件路径')

    args = parser.parse_args()

    # 创建临时args对象用于加载数据集
    class TempArgs:
        def __init__(self, dataset, datadir):
            self.dataset = dataset
            self.datadir = datadir

    temp_args = TempArgs(args.dataset, args.datadir)

    print(f"正在加载数据集: {args.dataset}")
    try:
        dataset = TUsDataset(temp_args)
        print(f"数据集加载成功!")
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return

    # 计算平均节点数
    avg_nodes = calculate_avg_nodes(dataset)
    print(f"数据集 {args.dataset} 的平均节点数: {avg_nodes}")

    # 生成触发器
    print(f"生成触发器参数:")
    print(f"  - 触发器节点数: {int(avg_nodes * args.frac_of_avg)} (avg_nodes={avg_nodes}, frac_of_avg={args.frac_of_avg})")
    print(f"  - 边密度: {args.density}")

    trigger = generate_trigger(avg_nodes, args.frac_of_avg, args.density)

    # 构造保存路径，按照与dis_bkd_fedgnn.py相同的格式
    filename = f"./Data/global_trigger/{args.seed}/{args.model}_{args.dataset}_{args.num_workers}_{args.num_mali}_{args.epoch_backdoor}_{args.frac_of_avg:.2f}_{args.poisoning_intensity:.2f}_{args.density:.2f}.pkl"

    # 保存触发器
    save_trigger(trigger, filename)

    print("\n触发器信息:")
    print(f"  - 节点数: {trigger.number_of_nodes()}")
    print(f"  - 边数: {trigger.number_of_edges()}")
    print(f"  - 实际边密度: {nx.density(trigger):.3f}")


if __name__ == '__main__':
    # 批量生成三个触发器
    configurations = [
        {'model': 'GCN', 'dataset': 'NCI1'},
        {'model': 'GCN', 'dataset': 'PROTEINS_full'},
        {'model': 'GAT', 'dataset': 'PROTEINS_full'},
    ]

    print("=" * 60)
    print("批量生成触发器")
    print("=" * 60)
    print()

    for config in configurations:
        # 模拟命令行参数
        import sys
        original_argv = sys.argv
        sys.argv = ['generate_trigger.py',
                   '--dataset', config['dataset'],
                   '--model', config['model']]

        try:
            main()
        except Exception as e:
            print(f"生成 {config['model']}_{config['dataset']} 触发器时出错: {e}")
        finally:
            sys.argv = original_argv
            print("-" * 60)
            print()

    print("所有触发器生成完成!")
