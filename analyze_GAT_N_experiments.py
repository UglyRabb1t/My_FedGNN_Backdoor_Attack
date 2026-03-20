"""
GAT_N_xxx 实验结果分析
实验配置：50轮攻击 + 50轮良性训练
实验目的：延长后门攻击的寿命
"""

import numpy as np
import os
import sys

# 输出文件路径
OUTPUT_FILE = 'GAT_N_analysis_results.txt'

# 保存原始stdout
original_stdout = sys.stdout

# 重定向输出到文件
sys.stdout = open(OUTPUT_FILE, 'w', encoding='utf-8')

# 实验配置
configs = {
    'Baseline': 'GAT_N_B_50_100',
    'r50': 'GAT_N_G_50_100_r50',
    'r60': 'GAT_N_G_50_100_r60',
    'r70': 'GAT_N_G_50_100_r70',
    'r80': 'GAT_N_G_50_100_r80',
    'r90': 'GAT_N_G_50_100_r90',
    'r95': 'GAT_N_G_50_100_r95',
    'r98': 'GAT_N_G_50_100_r98',
    'r99': 'GAT_N_G_50_100_r99',
}

ratios = {
    'Baseline': 1.0,
    'r50': 0.5,
    'r60': 0.6,
    'r70': 0.7,
    'r80': 0.8,
    'r90': 0.9,
    'r95': 0.95,
    'r98': 0.98,
    'r99': 0.99,
}

def parse_file(filepath):
    """读取单列数据文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return np.array([float(x.strip()) for x in lines if x.strip()])

def parse_attack_file(filepath):
    """读取攻击成功率文件（两列：全局ASR, 本地ASR）"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        if line.strip():
            parts = line.split()
            data.append([float(x) for x in parts])
    return np.array(data)

def compute_asr_decay_metrics(asr, attack_end=50):
    """
    计算ASR衰减相关指标
    """
    # 攻击期 ASR
    attack_asr = asr[:attack_end]

    # 持久期 ASR
    persist_asr = asr[attack_end:]

    # 基本统计
    attack_avg = attack_asr.mean()
    attack_std = attack_asr.std()
    attack_peak = attack_asr.max()

    persist_avg = persist_asr.mean()
    persist_std = persist_asr.std()
    persist_min = persist_asr.min()
    persist_final_last10 = persist_asr[-10:].mean()  # 最后10轮平均

    # 衰减指标
    decay_rate = (attack_avg - persist_avg) / attack_avg if attack_avg > 0 else 0
    decay_from_peak = (attack_peak - persist_avg) / attack_peak if attack_peak > 0 else 0
    final_decay = (attack_avg - persist_final_last10) / attack_avg if attack_avg > 0 else 0

    # 半衰期：ASR 降到攻击期平均值的50%时的epoch
    half_asr = attack_avg * 0.5
    half_life_epoch = None
    for i, val in enumerate(asr):
        if val < half_asr and i >= attack_end:  # 只在持久期找
            half_life_epoch = i + 1
            break
    if half_life_epoch is None:
        half_life_epoch = 100  # 如果没有降到50%，标记为100

    # 持久期超过25%阈值的比例
    threshold_25 = attack_avg * 0.25
    above_25_ratio = np.sum(persist_asr > threshold_25) / len(persist_asr) if len(persist_asr) > 0 else 0

    # 超过10%阈值的比例
    threshold_10 = attack_avg * 0.10
    above_10_ratio = np.sum(persist_asr > threshold_10) / len(persist_asr) if len(persist_asr) > 0 else 0

    # 衰减拟合斜率（持久期线性拟合）
    if len(persist_asr) > 1:
        x = np.arange(len(persist_asr))
        slope, intercept = np.polyfit(x, persist_asr, 1)
    else:
        slope = 0
        intercept = persist_avg

    return {
        'attack_avg': attack_avg,
        'attack_std': attack_std,
        'attack_peak': attack_peak,
        'persist_avg': persist_avg,
        'persist_std': persist_std,
        'persist_min': persist_min,
        'persist_final_last10': persist_final_last10,
        'decay_rate': decay_rate,
        'decay_from_peak': decay_from_peak,
        'final_decay': final_decay,
        'half_life_epoch': half_life_epoch,
        'above_25_ratio': above_25_ratio,
        'above_10_ratio': above_10_ratio,
        'decay_slope': slope,
    }

def compute_volatility_metrics(asr, attack_end=50):
    """
    计算波动性指标
    """
    persist_asr = asr[attack_end:]

    # 基本波动指标
    std = persist_asr.std()
    mean = persist_asr.mean()
    cv = std / mean if mean > 0 else 0  # 变异系数
    range_val = persist_asr.max() - persist_asr.min()  # 极差

    # 移动标准差（窗口大小=5）
    if len(persist_asr) >= 5:
        moving_std = []
        for i in range(len(persist_asr) - 4):
            moving_std.append(persist_asr[i:i+5].std())
        avg_moving_std = np.mean(moving_std)
    else:
        avg_moving_std = 0

    # 相邻差分的绝对值（衡量短期波动）
    if len(persist_asr) > 1:
        diff_abs = np.abs(np.diff(persist_asr))
        avg_diff = diff_abs.mean()
        max_diff = diff_abs.max()
    else:
        avg_diff = 0
        max_diff = 0

    # 局部波动峰值（超过2倍标准差的点）
    if std > 0:
        outliers = np.sum(np.abs(persist_asr - mean) > 2 * std)
    else:
        outliers = 0

    return {
        'persist_std': std,
        'persist_cv': cv,
        'persist_range': range_val,
        'avg_moving_std': avg_moving_std,
        'avg_diff': avg_diff,
        'max_diff': max_diff,
        'outliers': outliers,
    }

def compute_benign_metrics(test_acc, attack_end=50):
    """
    计算良性准确率指标
    """
    persist_acc = test_acc[attack_end:]

    persist_avg = persist_acc.mean()
    persist_std = persist_acc.std()
    persist_peak = persist_acc.max()
    persist_final_last10 = persist_acc[-10:].mean()

    # 从攻击期到持久期的提升
    attack_acc = test_acc[:attack_end]
    improvement = persist_avg - attack_acc.mean()
    improvement_rate = improvement / attack_acc.mean() if attack_acc.mean() > 0 else 0

    return {
        'persist_avg': persist_avg,
        'persist_std': persist_std,
        'persist_peak': persist_peak,
        'persist_final_last10': persist_final_last10,
        'improvement': improvement,
        'improvement_rate': improvement_rate,
    }

# ============================================================
# 主分析函数
# ============================================================

def analyze_all():
    """分析所有实验结果"""

    print("=" * 100)
    print("GAT_N_xxx 实验结果分析 - 后门攻击寿命研究")
    print("实验配置: 50轮攻击 + 50轮良性训练")
    print("=" * 100)
    print()

    # 存储所有结果
    all_results = {}

    for name, dir_name in configs.items():
        # 读取数据
        test_file = f'Results/{dir_name}/0/GAT_NCI1_5_1_0.20_0.20_0.80_global_test.txt'
        attack_file = f'Results/{dir_name}/0/GAT_NCI1_5_1_0.20_0.20_0.80_global_attack.txt'

        if not os.path.exists(test_file) or not os.path.exists(attack_file):
            print(f"警告: {name} 的数据文件不完整，跳过")
            continue

        test_acc = parse_file(test_file)
        attack_data = parse_attack_file(attack_file)
        global_asr = attack_data[:, 0]  # 全局攻击成功率

        # 计算指标
        asr_metrics = compute_asr_decay_metrics(global_asr)
        volatility_metrics = compute_volatility_metrics(global_asr)
        benign_metrics = compute_benign_metrics(test_acc)

        all_results[name] = {
            'ratio': ratios[name],
            'asr_metrics': asr_metrics,
            'volatility_metrics': volatility_metrics,
            'benign_metrics': benign_metrics,
        }

    # ============================================================
    # 输出结果
    # ============================================================

    print("=" * 100)
    print("【1】后门攻击寿命分析 - ASR 衰减指标")
    print("=" * 100)
    print()
    print(f"{'配置':<12} {'r值':<6} {'攻击期ASR':<12} {'持久期ASR':<12} {'衰减率':<10} {'最终ASR(最后10轮)':<18} {'半衰期':<8}")
    print("-" * 100)

    for name in configs.keys():
        if name not in all_results:
            continue
        r = all_results[name]['ratio']
        m = all_results[name]['asr_metrics']
        print(f"{name:<12} {r:<6.2f} {m['attack_avg']:<12.4f} {m['persist_avg']:<12.4f} "
              f"{m['decay_rate']:<10.2%} {m['persist_final_last10']:<18.4f} {m['half_life_epoch']:<8d}")

    print()
    print(f"{'配置':<12} {'r值':<6} {'从峰值衰减':<12} {'从均值最终衰减':<16} {'超过25%阈值比例':<16} {'超过10%阈值比例':<16} {'衰减斜率':<12}")
    print("-" * 100)

    for name in configs.keys():
        if name not in all_results:
            continue
        r = all_results[name]['ratio']
        m = all_results[name]['asr_metrics']
        print(f"{name:<12} {r:<6.2f} {m['decay_from_peak']:<12.2%} {m['final_decay']:<16.2%} "
              f"{m['above_25_ratio']:<16.2%} {m['above_10_ratio']:<16.2%} {m['decay_slope']:<12.4f}")

    print()
    print("=" * 100)
    print("【2】波动性分析 - 确认 r 较大时的波动情况")
    print("=" * 100)
    print()
    print(f"{'配置':<12} {'r值':<6} {'持久期标准差':<14} {'变异系数':<14} {'极差':<10} {'平均移动标准差':<18} {'平均相邻差':<14} {'最大相邻差':<14} {'离群点数':<10}")
    print("-" * 140)

    for name in configs.keys():
        if name not in all_results:
            continue
        r = all_results[name]['ratio']
        v = all_results[name]['volatility_metrics']
        print(f"{name:<12} {r:<6.2f} {v['persist_std']:<14.4f} {v['persist_cv']:<14.4f} "
              f"{v['persist_range']:<10.4f} {v['avg_moving_std']:<18.4f} {v['avg_diff']:<14.4f} "
              f"{v['max_diff']:<14.4f} {v['outliers']:<10d}")

    print()
    print("=" * 100)
    print("【3】良性准确率分析")
    print("=" * 100)
    print()
    print(f"{'配置':<12} {'r值':<6} {'持久期平均准确率':<18} {'持久期最高准确率':<18} {'最终准确率(最后10轮)':<22} {'提升幅度':<12} {'提升率':<10}")
    print("-" * 110)

    for name in configs.keys():
        if name not in all_results:
            continue
        r = all_results[name]['ratio']
        b = all_results[name]['benign_metrics']
        print(f"{name:<12} {r:<6.2f} {b['persist_avg']:<18.4f} {b['persist_peak']:<18.4f} "
              f"{b['persist_final_last10']:<22.4f} {b['improvement']:<12.4f} {b['improvement_rate']:<10.2%}")

    print()
    print("=" * 100)
    print("【4】r 与 波动性 的关系分析")
    print("=" * 100)
    print()

    # 提取 r 和波动性指标
    r_values = []
    std_values = []
    cv_values = []
    avg_diff_values = []

    for name in configs.keys():
        if name not in all_results:
            continue
        r_values.append(all_results[name]['ratio'])
        std_values.append(all_results[name]['volatility_metrics']['persist_std'])
        cv_values.append(all_results[name]['volatility_metrics']['persist_cv'])
        avg_diff_values.append(all_results[name]['volatility_metrics']['avg_diff'])

    # 简单相关分析
    r_arr = np.array(r_values)
    std_arr = np.array(std_values)
    cv_arr = np.array(cv_values)
    diff_arr = np.array(avg_diff_values)

    corr_r_std = np.corrcoef(r_arr, std_arr)[0, 1]
    corr_r_cv = np.corrcoef(r_arr, cv_arr)[0, 1]
    corr_r_diff = np.corrcoef(r_arr, diff_arr)[0, 1]

    print(f"r 与 持久期标准差 的相关系数: {corr_r_std:.4f}")
    print(f"r 与 变异系数(CV) 的相关系数: {corr_r_cv:.4f}")
    print(f"r 与 平均相邻差 的相关系数: {corr_r_diff:.4f}")
    print()
    if corr_r_std > 0.5:
        print("结论: r 与 波动性 呈正相关，r 越大波动性越大！")
    elif corr_r_std < -0.5:
        print("结论: r 与 波动性 呈负相关，r 越小波动性越大！")
    else:
        print("结论: r 与 波动性 的相关性较弱或不明显。")

    print()
    print("=" * 100)
    print("【5】最优配置推荐（综合考虑后门寿命和稳定性）")
    print("=" * 100)
    print()

    # 综合评分计算
    scores = {}
    for name in configs.keys():
        if name not in all_results:
            continue

        r = all_results[name]['ratio']
        asr_m = all_results[name]['asr_metrics']
        vol_m = all_results[name]['volatility_metrics']
        benign_m = all_results[name]['benign_metrics']

        # 评分标准（权重可调整）:
        # 1. 持久期ASR越高越好 (权重: 40%)
        asr_score = asr_m['persist_avg'] / 1.0  # 归一化

        # 2. 衰减率越低越好 (权重: 30%)
        decay_score = 1 - asr_m['decay_rate']

        # 3. 波动性越低越好 (权重: 20%)
        # 使用变异系数的倒数
        volatility_score = 1 / (1 + vol_m['persist_cv']) if vol_m['persist_cv'] > 0 else 1

        # 4. 良性准确率越高越好 (权重: 10%)
        benign_score = benign_m['persist_avg'] / 1.0

        # 综合得分
        total_score = (
            asr_score * 0.40 +
            decay_score * 0.30 +
            volatility_score * 0.20 +
            benign_score * 0.10
        )

        scores[name] = total_score

    # 排序输出
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"{'排名':<6} {'配置':<12} {'r值':<8} {'综合得分':<12} {'持久期ASR':<12} {'衰减率':<12} {'波动CV':<10}")
    print("-" * 80)

    for rank, (name, score) in enumerate(sorted_scores, 1):
        r = all_results[name]['ratio']
        asr_m = all_results[name]['asr_metrics']
        vol_m = all_results[name]['volatility_metrics']
        print(f"{rank:<6} {name:<12} {r:<8.2f} {score:<12.4f} {asr_m['persist_avg']:<12.4f} "
              f"{asr_m['decay_rate']:<12.2%} {vol_m['persist_cv']:<10.4f}")

    print()

    return all_results

if __name__ == "__main__":
    results = analyze_all()

    # 恢复原始 stdout 并关闭文件
    sys.stdout.close()
    sys.stdout = original_stdout

    print(f"分析完成！结果已保存至: {OUTPUT_FILE}")
