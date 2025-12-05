import os
import sys

os.environ['ARROW_PRE_0_15_IPC_FORMAT'] = '1'

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置pandas优先使用fastparquet
try:
    import fastparquet
    # 确保pandas使用fastparquet作为默认引擎
    pd.options.io.parquet.engine = 'fastparquet'
except ImportError:
    print("Warning: fastparquet not available, may use pyarrow")

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data.feature_engineering import FeatureEngineeringPipeline
from src.data.selection import calculate_iv, calculate_psi
from src.data.loader import load_parquet_data
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_feature_selection_results(train_features, y_train, selected_features, max_features=None):
    """分析特征选择结果"""
    print("\n" + "=" * 60)
    print("特征选择结果分析")
    print("=" * 60)

    # 如果没有标签，无法计算IV，直接返回
    if y_train is None:
        print("警告: y_train 为 None，无法计算IV，跳过特征选择结果分析。")
        return {}, []
    
    # 计算特征的IV值
    feature_cols = [col for col in selected_features if col in train_features.columns]
    if not feature_cols:
        print("警告: 没有可用于计算IV的特征。")
        return {}, []
    if max_features:
        feature_cols = feature_cols[:max_features]

    # 检查标签是否包含两类
    unique_classes = np.unique(y_train[~pd.isna(y_train)])
    if len(unique_classes) < 2:
        raise ValueError("y_train 中只有单一类别，无法计算IV。请增大样本或关闭采样以包含正负样本。")
    iv_scores = {}
    
    print("\n计算特征IV值...")
    for col in feature_cols:
        try:
            df_temp = pd.DataFrame({col: train_features[col], 'target': y_train})
            iv = calculate_iv(df_temp, col, 'target')
            iv_scores[col] = iv
        except:
            iv_scores[col] = 0.0
    
    # 排序
    iv_sorted = sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n总特征数: {len(feature_cols)}")
    print(f"选中特征数: {len(selected_features)}")
    print(f"特征压缩率: {len(selected_features)/len(feature_cols)*100:.2f}%")
    
    print("\nTop 20 特征IV值:")
    for i, (feat, iv) in enumerate(iv_sorted[:20], 1):
        status = "[SELECTED]" if feat in selected_features else "[NOT SELECTED]"
        print(f"{i:2d}. {status} {feat:40s} IV={iv:.4f}")
    
    return iv_scores, iv_sorted

def analyze_feature_categories(selected_features):
    """分析特征类别分布"""
    print("\n" + "=" * 60)
    print("特征类别分析")
    print("=" * 60)
    
    categories = {
        '时窗聚合': [f for f in selected_features if 'avg_last' in f or 'std_last' in f],
        '趋势斜率': [f for f in selected_features if 'trend' in f or 'slope' in f],
        '统计聚合': [f for f in selected_features if any(x in f for x in ['_mean', '_std', '_min', '_max', '_last'])],
        '时序嵌入': [f for f in selected_features if 'emb' in f or 'transformer' in f or 'gru' in f],
        '缺失指示': [f for f in selected_features if 'is_missing' in f],
        '异常指示': [f for f in selected_features if 'is_outlier' in f],
        '其他': []
    }
    
    # 分类其他特征
    categorized = set()
    for cat, feats in categories.items():
        if cat != '其他':
            categorized.update(feats)
    
    categories['其他'] = [f for f in selected_features if f not in categorized]
    
    print("\n特征类别分布:")
    total_selected = max(1, len(selected_features))
    for cat, feats in categories.items():
        print(f"  {cat:12s}: {len(feats):4d} 个特征 ({len(feats)/total_selected*100:5.2f}%)")
        if len(feats) > 0 and len(feats) <= 10:
            print(f"    示例: {', '.join(feats[:5])}")
    
    return categories

def visualize_feature_importance(iv_sorted, selected_features, output_dir='results/figures'):
    """可视化特征重要性"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not iv_sorted:
        print("  跳过特征重要性图（无IV数据）")
        return
    
    # 提取数据
    features = [f[0] for f in iv_sorted[:30]]
    iv_values = [f[1] for f in iv_sorted[:30]]
    is_selected = [f in selected_features for f in features]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['green' if sel else 'red' for sel in is_selected]
    
    bars = ax.barh(range(len(features)), iv_values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel('信息价值 (IV)', fontsize=12)
    ax.set_title('Top 30 特征信息价值分布（绿色=选中，红色=未选中）', fontsize=14)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_3_6_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 特征重要性图已保存: {output_dir}/fig_3_6_feature_importance.png")
    plt.close()

def visualize_feature_categories(categories, output_dir='results/figures'):
    """可视化特征类别分布"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 过滤掉空类别
    categories_filtered = {k: v for k, v in categories.items() if len(v) > 0}
    if not categories_filtered:
        print("  跳过特征类别分布图（无类别数据）")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 饼图
    labels = list(categories_filtered.keys())
    sizes = [len(categories_filtered[k]) for k in labels]
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_pie)
    ax1.set_title('特征类别分布（饼图）', fontsize=14)
    
    # 柱状图
    bars = ax2.barh(range(len(labels)), sizes, color=colors_pie)
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel('特征数量', fontsize=12)
    ax2.set_title('特征类别分布（柱状图）', fontsize=14)
    ax2.grid(axis='x', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for i, (bar, size) in enumerate(zip(bars, sizes)):
        ax2.text(size + max(sizes) * 0.01, i, f'{size}', 
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_feature_categories.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 特征类别分布图已保存: {output_dir}/fig_feature_categories.png")
    plt.close()

def visualize_feature_correlation(train_features, selected_features, output_dir='results/figures', max_features=30):
    """可视化特征相关性热力图"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 选择要可视化的特征（选择前N个）
    features_to_plot = selected_features[:max_features]
    features_to_plot = [f for f in features_to_plot if f in train_features.columns]
    
    if len(features_to_plot) < 2:
        print("  跳过特征相关性图（特征数量不足）")
        return
    
    # 计算相关性矩阵
    corr_matrix = train_features[features_to_plot].corr()
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # 只显示下三角
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                xticklabels=[f[:20] for f in features_to_plot],  # 截断长名称
                yticklabels=[f[:20] for f in features_to_plot],
                ax=ax)
    
    ax.set_title(f'Top {len(features_to_plot)} 特征相关性热力图', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_feature_correlation.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 特征相关性热力图已保存: {output_dir}/fig_feature_correlation.png")
    plt.close()

def visualize_feature_statistics(train_features, selected_features, y_train=None, output_dir='results/figures'):
    """可视化特征统计信息"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    features_to_plot = [f for f in selected_features[:20] if f in train_features.columns]
    if not features_to_plot:
        print("  跳过特征统计图（无可用特征）")
        return
    
    # 计算统计信息
    stats_data = []
    for feat in features_to_plot:
        data = train_features[feat].dropna()
        if len(data) > 0:
            stats_data.append({
                'feature': feat[:30],  # 截断长名称
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max(),
                'median': data.median()
            })
    
    if not stats_data:
        print("  跳过特征统计图（无有效数据）")
        return
    
    stats_df = pd.DataFrame(stats_data)
    
    # 创建多子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 均值分布
    axes[0, 0].barh(range(len(stats_df)), stats_df['mean'], alpha=0.7, color='skyblue')
    axes[0, 0].set_yticks(range(len(stats_df)))
    axes[0, 0].set_yticklabels(stats_df['feature'], fontsize=8)
    axes[0, 0].set_xlabel('均值', fontsize=10)
    axes[0, 0].set_title('特征均值分布', fontsize=12)
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # 2. 标准差分布
    axes[0, 1].barh(range(len(stats_df)), stats_df['std'], alpha=0.7, color='lightcoral')
    axes[0, 1].set_yticks(range(len(stats_df)))
    axes[0, 1].set_yticklabels(stats_df['feature'], fontsize=8)
    axes[0, 1].set_xlabel('标准差', fontsize=10)
    axes[0, 1].set_title('特征标准差分布', fontsize=12)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # 3. 值域分布（max - min）
    value_range = stats_df['max'] - stats_df['min']
    axes[1, 0].barh(range(len(stats_df)), value_range, alpha=0.7, color='lightgreen')
    axes[1, 0].set_yticks(range(len(stats_df)))
    axes[1, 0].set_yticklabels(stats_df['feature'], fontsize=8)
    axes[1, 0].set_xlabel('值域 (Max - Min)', fontsize=10)
    axes[1, 0].set_title('特征值域分布', fontsize=12)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 4. 如果有标签，显示不同类别下的特征分布对比
    if y_train is not None and len(np.unique(y_train[~pd.isna(y_train)])) >= 2:
        # 选择第一个特征进行对比
        feat = features_to_plot[0]
        data_0 = train_features.loc[y_train == 0, feat].dropna()
        data_1 = train_features.loc[y_train == 1, feat].dropna()
        
        if len(data_0) > 0 and len(data_1) > 0:
            axes[1, 1].hist(data_0, bins=30, alpha=0.6, label='类别 0', color='blue', density=True)
            axes[1, 1].hist(data_1, bins=30, alpha=0.6, label='类别 1', color='red', density=True)
            axes[1, 1].set_xlabel(f'{feat[:30]}', fontsize=10)
            axes[1, 1].set_ylabel('密度', fontsize=10)
            axes[1, 1].set_title(f'特征 {feat[:30]} 在不同类别下的分布', fontsize=12)
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, '数据不足', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
    else:
        # 显示中位数分布
        axes[1, 1].barh(range(len(stats_df)), stats_df['median'], alpha=0.7, color='orange')
        axes[1, 1].set_yticks(range(len(stats_df)))
        axes[1, 1].set_yticklabels(stats_df['feature'], fontsize=8)
        axes[1, 1].set_xlabel('中位数', fontsize=10)
        axes[1, 1].set_title('特征中位数分布', fontsize=12)
        axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.suptitle('特征统计信息可视化', fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(f'{output_dir}/fig_feature_statistics.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 特征统计信息图已保存: {output_dir}/fig_feature_statistics.png")
    plt.close()

def visualize_iv_distribution(iv_scores, output_dir='results/figures'):
    """可视化IV值分布"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if not iv_scores:
        print("  跳过IV分布图（无IV数据）")
        return
    
    iv_values = list(iv_scores.values())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 直方图
    ax1.hist(iv_values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(np.mean(iv_values), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(iv_values):.3f}')
    ax1.axvline(np.median(iv_values), color='green', linestyle='--', linewidth=2, label=f'中位数: {np.median(iv_values):.3f}')
    ax1.set_xlabel('IV值', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('IV值分布直方图', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 箱线图
    ax2.boxplot(iv_values, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_ylabel('IV值', fontsize=12)
    ax2.set_title('IV值箱线图', fontsize=14)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_iv_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ IV值分布图已保存: {output_dir}/fig_iv_distribution.png")
    plt.close()

def generate_all_visualizations(train_features, y_train, selected_features, 
                                iv_scores, iv_sorted, categories, output_dir='results/figures'):
    """生成所有可视化图表"""
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 特征重要性图（如果有IV）
    if iv_sorted:
        visualize_feature_importance(iv_sorted, selected_features, output_dir)
    
    # 2. 特征类别分布图
    visualize_feature_categories(categories, output_dir)
    
    # 3. 特征相关性热力图
    visualize_feature_correlation(train_features, selected_features, output_dir)
    
    # 4. 特征统计信息图
    visualize_feature_statistics(train_features, selected_features, y_train, output_dir)
    
    # 5. IV值分布图（如果有IV）
    if iv_scores:
        visualize_iv_distribution(iv_scores, output_dir)
    
    print("\n" + "=" * 60)
    print(f"所有可视化图表已保存到: {output_dir}")
    print("=" * 60)

def main():
    """主函数"""
    print("=" * 80)
    print("特征工程流程执行与结果分析")
    print("=" * 80)
    
    # 配置
    sample_size = 50000  # 使用采样以加快速度
    output_dir = Path('results/feature_engineering')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 运行特征工程流程
    print("\n步骤1: 运行特征工程流程...")
    pipeline = FeatureEngineeringPipeline(dataset_type='amex')
    
    # 运行完整特征工程管道
    train_features, test_features, y_train = pipeline.run_full_pipeline(
        train_path='data/raw/train.parquet',
        test_path='data/raw/test.parquet',
        target_col='target',
        sample_size=sample_size,
        build_sequential=False,  # 暂时跳过时序特征以加快速度
        select_features=True
    )
    
    # 保存结果
    train_features.to_parquet(output_dir / 'train_features.parquet', index=False)
    print(f"\n特征已保存到: {output_dir}")
    
    # 2. 分析特征选择结果
    print("\n步骤2: 分析特征选择结果...")
    selected_features = pipeline.selected_features
    if selected_features is None:
        # 如果没有进行特征选择，使用所有特征
        selected_features = [col for col in train_features.columns if col != 'customer_ID']
    
    # 如果没有标签，只能基于特征本身做统计分析，跳过IV计算
    if y_train is None:
        print("警告: 训练数据中没有 target 列，无法计算IV并做真正的特征选择，将跳过IV分析，仅对特征做统计分析。")
        iv_scores, iv_sorted = {}, []
    else:
        iv_scores, iv_sorted = analyze_feature_selection_results(
            train_features, y_train, selected_features
        )
    
    # 3. 分析特征类别
    categories = analyze_feature_categories(selected_features)
    
    # 4. 生成所有可视化图表
    print("\n步骤3: 生成可视化图表...")
    generate_all_visualizations(
        train_features, y_train, selected_features,
        iv_scores, iv_sorted, categories, 
        output_dir='results/figures'
    )
    
    # 5. 生成统计报告（报告中会根据 iv_scores 是否为空自动处理）
    print("\n步骤4: 生成统计报告...")
    generate_statistics_report(
        train_features, y_train, selected_features, 
        iv_scores, categories, output_dir
    )
    
    print("\n" + "=" * 80)
    print("特征工程分析完成！")
    print("=" * 80)
    
    return train_features, y_train, selected_features, iv_scores, categories

def generate_statistics_report(train_features, y_train, selected_features, 
                              iv_scores, categories, output_dir):
    """生成统计报告"""
    report = []
    report.append("=" * 80)
    report.append("特征工程统计报告")
    report.append("=" * 80)
    report.append("")
    
    # 基本信息
    if train_features is not None:
        total_features = len([c for c in train_features.columns if c != 'customer_ID'])
        compression_rate = len(selected_features)/total_features*100 if total_features > 0 else 0
    else:
        total_features = len(selected_features)  # 如果train_features为None，使用selected_features的长度
        compression_rate = 100.0
    
    report.append(f"总特征数: {total_features}")
    report.append(f"选中特征数: {len(selected_features)}")
    report.append(f"特征压缩率: {compression_rate:.2f}%")
    report.append("")
    
    # 特征类别分布
    report.append("特征类别分布:")
    total_selected = max(1, len(selected_features))
    for cat, feats in categories.items():
        report.append(f"  {cat:12s}: {len(feats):4d} 个特征 ({len(feats)/total_selected*100:5.2f}%)")
    report.append("")
    
    # Top特征
    report.append("Top 20 重要特征 (按IV值排序):")
    sorted_features = sorted(selected_features, 
                            key=lambda x: iv_scores.get(x, 0), 
                            reverse=True)
    for i, feat in enumerate(sorted_features[:20], 1):
        iv = iv_scores.get(feat, 0)
        report.append(f"  {i:2d}. {feat:50s} IV={iv:.4f}")
    
    # 保存报告
    report_path = output_dir / 'feature_statistics_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n统计报告已保存: {report_path}")
    print('\n'.join(report))

    """生成模拟结果（用于演示）"""
    print("生成模拟特征工程结果...")
    
    # 模拟特征选择结果
    selected_features = [
        'P_2_mean', 'P_2_std', 'P_2_trend_3m_slope',
        'B_1_mean', 'B_1_std', 'B_1_avg_last_3m',
        'D_39_mean', 'D_39_trend_6m_slope',
        'S_2_mean', 'S_2_std', 'S_2_avg_last_6m',
        'P_2_is_missing', 'B_1_is_outlier',
        'transformer_emb_0', 'transformer_emb_1', 'transformer_emb_2'
    ]
    
    # 模拟IV值
    iv_scores = {
        'P_2_mean': 0.35, 'P_2_std': 0.28, 'P_2_trend_3m_slope': 0.42,
        'B_1_mean': 0.31, 'B_1_std': 0.25, 'B_1_avg_last_3m': 0.38,
        'D_39_mean': 0.29, 'D_39_trend_6m_slope': 0.33,
        'S_2_mean': 0.27, 'S_2_std': 0.23, 'S_2_avg_last_6m': 0.30,
        'P_2_is_missing': 0.18, 'B_1_is_outlier': 0.15,
        'transformer_emb_0': 0.32, 'transformer_emb_1': 0.28, 'transformer_emb_2': 0.25
    }
    
    iv_sorted = sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 分析特征类别
    categories = analyze_feature_categories(selected_features)
    
    # 可视化
    visualize_feature_importance(iv_sorted, selected_features)
    
    # 生成报告
    generate_statistics_report(
        None, None, selected_features, 
        iv_scores, categories, output_dir
    )
    
    return None, None, selected_features, iv_scores, categories

if __name__ == '__main__':
    main()

