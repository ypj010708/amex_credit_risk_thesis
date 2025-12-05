"""
ATAE-GAN 训练结果可视化脚本
生成训练损失曲线、真实vs生成样本分布对比等图表
"""
import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from pathlib import Path

def _set_chinese_font():
    """
    根据当前系统可用字体，自动设置支持中文的字体。
    改进版：更强大的字体检测和加载机制
    """
    # 首先设置全局配置
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检测操作系统
    import platform
    system = platform.system()
    
    print(f"\n检测到操作系统: {system}")
    
    # 根据系统选择字体策略
    if system == 'Windows':
        fonts_to_try = [
            'Microsoft YaHei',
            'SimHei', 
            'SimSun',
            'KaiTi',
            'FangSong'
        ]
        font_paths = [
            r'C:\Windows\Fonts\msyh.ttc',
            r'C:\Windows\Fonts\msyhbd.ttc',
            r'C:\Windows\Fonts\simhei.ttf',
            r'C:\Windows\Fonts\simsun.ttc',
        ]
    elif system == 'Darwin':  # macOS
        fonts_to_try = [
            'PingFang SC',
            'Heiti SC',
            'STHeiti',
            'Arial Unicode MS'
        ]
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/Supplemental/Songti.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
        ]
    else:  # Linux
        fonts_to_try = [
            'WenQuanYi Micro Hei',
            'WenQuanYi Zen Hei',
            'Noto Sans CJK SC',
            'Droid Sans Fallback'
        ]
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
        ]
    
    # 尝试从已安装的字体中查找
    available_fonts = set([f.name for f in font_manager.fontManager.ttflist])
    
    print("正在查找中文字体...")
    font_found = False
    for font_name in fonts_to_try:
        if font_name in available_fonts:
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
            print(f"✓ 成功使用字体: {font_name}")
            font_found = True
            break
        else:
            print(f"  × 字体不可用: {font_name}")
    
    # 如果没找到，尝试手动加载字体文件
    if not font_found:
        print("\n尝试从文件路径加载字体...")
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font_manager.fontManager.addfont(font_path)
                    # 重新获取字体名称
                    for font in font_manager.fontManager.ttflist:
                        if font.fname == font_path:
                            font_name = font.name
                            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                            print(f"✓ 成功加载字体文件: {font_path}")
                            print(f"✓ 使用字体: {font_name}")
                            font_found = True
                            break
                    if font_found:
                        break
                except Exception as e:
                    print(f"  × 加载失败: {font_path} - {e}")
                    continue
            else:
                print(f"  × 文件不存在: {font_path}")
    
    if not font_found:
        print("\n⚠ 警告: 未找到合适的中文字体!")
        print("解决方案:")
        print("  1. Windows: 确保安装了微软雅黑字体")
        print("  2. macOS: 系统自带苹方字体")
        print("  3. Linux: 运行 sudo apt-get install fonts-wqy-microhei")
        print("  4. 或手动指定字体: plt.rcParams['font.sans-serif'] = ['字体名']")
        # 最后的备用方案
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    
    # 设置样式
    sns.set_style('whitegrid')
    
    # 清除字体缓存（有时需要）
    try:
        font_manager._rebuild()
    except:
        pass

# 在导入后立即设置字体
_set_chinese_font()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

def visualize_training_history(trainer, output_dir='results/atae_gan/figures'):
    """可视化训练损失曲线"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    history = trainer.history
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 自编码器预训练损失
    if history['pretrain_loss']:
        axes[0, 0].plot(history['pretrain_loss'], label='重构损失', color='blue', linewidth=2)
        axes[0, 0].set_xlabel('训练轮次 (Epoch)', fontsize=12)
        axes[0, 0].set_ylabel('均方误差损失', fontsize=12)
        axes[0, 0].set_title('ATAE 自编码器预训练损失', fontsize=14, pad=10)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 判别器损失
    if history['critic_loss']:
        axes[0, 1].plot(history['critic_loss'], label='判别器损失', color='red', linewidth=2)
        axes[0, 1].set_xlabel('训练轮次 (Epoch)', fontsize=12)
        axes[0, 1].set_ylabel('Wasserstein距离', fontsize=12)
        axes[0, 1].set_title('判别器 (Critic) 训练损失', fontsize=14, pad=10)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 生成器损失
    if history['generator_loss']:
        axes[1, 0].plot(history['generator_loss'], label='生成器损失', color='green', linewidth=2)
        axes[1, 0].set_xlabel('训练轮次 (Epoch)', fontsize=12)
        axes[1, 0].set_ylabel('损失值', fontsize=12)
        axes[1, 0].set_title('生成器训练损失', fontsize=14, pad=10)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 分类器反馈损失（如果有）
    if history['classifier_loss'] and len(history['classifier_loss']) > 0:
        axes[1, 1].plot(history['classifier_loss'], label='分类器反馈损失', color='orange', linewidth=2)
        axes[1, 1].set_xlabel('训练轮次 (Epoch)', fontsize=12)
        axes[1, 1].set_ylabel('均方误差损失', fontsize=12)
        axes[1, 1].set_title('分类器反馈损失', fontsize=14, pad=10)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, '分类器反馈未启用', 
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].axis('off')
    
    plt.suptitle('ATAE-GAN 训练损失曲线', fontsize=18, y=0.995, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_history.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 训练损失曲线已保存: {output_dir}/training_history.png")
    plt.close()

def visualize_sample_distribution(real_samples, synthetic_samples, output_dir='results/atae_gan/figures', n_features=5):
    """对比真实样本和生成样本的分布"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 转换为numpy
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.cpu().numpy()
    if isinstance(synthetic_samples, torch.Tensor):
        synthetic_samples = synthetic_samples.cpu().numpy()
    
    # 随机选择几个特征进行可视化
    n_features = min(n_features, real_samples.shape[1])
    feature_indices = np.random.choice(real_samples.shape[1], n_features, replace=False)
    
    fig, axes = plt.subplots(2, n_features, figsize=(4*n_features, 8))
    if n_features == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, feat_idx in enumerate(feature_indices):
        real_feat = real_samples[:, feat_idx]
        synth_feat = synthetic_samples[:, feat_idx]
        
        # 上排：直方图对比
        axes[0, idx].hist(real_feat, bins=30, alpha=0.6, label='真实样本', color='blue', density=True)
        axes[0, idx].hist(synth_feat, bins=30, alpha=0.6, label='生成样本', color='red', density=True)
        axes[0, idx].set_title(f'特征 {feat_idx} 分布对比', fontsize=12, pad=8)
        axes[0, idx].set_xlabel('特征值', fontsize=10)
        axes[0, idx].set_ylabel('密度', fontsize=10)
        axes[0, idx].legend(fontsize=9)
        axes[0, idx].grid(True, alpha=0.3)
        
        # 下排：箱线图对比
        data_to_plot = [real_feat, synth_feat]
        bp = axes[1, idx].boxplot(data_to_plot, labels=['真实', '生成'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[1, idx].set_title(f'特征 {feat_idx} 箱线图', fontsize=12, pad=8)
        axes[1, idx].set_ylabel('特征值', fontsize=10)
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.suptitle('真实样本 vs 生成样本分布对比', fontsize=16, y=0.995, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sample_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 样本分布对比图已保存: {output_dir}/sample_distribution_comparison.png")
    plt.close()

def visualize_statistics_comparison(real_samples, synthetic_samples, output_dir='results/atae_gan/figures'):
    """对比真实和生成样本的统计特征"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.cpu().numpy()
    if isinstance(synthetic_samples, torch.Tensor):
        synthetic_samples = synthetic_samples.cpu().numpy()
    
    # 计算统计量
    real_mean = np.mean(real_samples, axis=0)
    real_std = np.std(real_samples, axis=0)
    synth_mean = np.mean(synthetic_samples, axis=0)
    synth_std = np.std(synthetic_samples, axis=0)
    
    # 选择前20个特征进行可视化
    n_features = min(20, real_samples.shape[1])
    feature_indices = np.arange(n_features)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 均值对比
    x_pos = np.arange(n_features)
    width = 0.35
    axes[0].bar(x_pos - width/2, real_mean[feature_indices], width, 
                label='真实样本', color='blue', alpha=0.7)
    axes[0].bar(x_pos + width/2, synth_mean[feature_indices], width,
                label='生成样本', color='red', alpha=0.7)
    axes[0].set_xlabel('特征索引', fontsize=12)
    axes[0].set_ylabel('均值', fontsize=12)
    axes[0].set_title('特征均值对比（前20个特征）', fontsize=14, pad=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 标准差对比
    axes[1].bar(x_pos - width/2, real_std[feature_indices], width,
                label='真实样本', color='blue', alpha=0.7)
    axes[1].bar(x_pos + width/2, synth_std[feature_indices], width,
                label='生成样本', color='red', alpha=0.7)
    axes[1].set_xlabel('特征索引', fontsize=12)
    axes[1].set_ylabel('标准差', fontsize=12)
    axes[1].set_title('特征标准差对比（前20个特征）', fontsize=14, pad=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/statistics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 统计特征对比图已保存: {output_dir}/statistics_comparison.png")
    plt.close()

def visualize_correlation_comparison(real_samples, synthetic_samples, output_dir='results/atae_gan/figures', max_features=20):
    """对比真实和生成样本的特征相关性"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if isinstance(real_samples, torch.Tensor):
        real_samples = real_samples.cpu().numpy()
    if isinstance(synthetic_samples, torch.Tensor):
        synthetic_samples = synthetic_samples.cpu().numpy()
    
    # 选择前N个特征
    n_features = min(max_features, real_samples.shape[1])
    real_subset = real_samples[:, :n_features]
    synth_subset = synthetic_samples[:, :n_features]
    
    # 计算相关性矩阵
    real_corr = np.corrcoef(real_subset.T)
    synth_corr = np.corrcoef(synth_subset.T)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 真实样本相关性
    im1 = axes[0].imshow(real_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_title('真实样本特征相关性矩阵', fontsize=14, pad=10)
    axes[0].set_xlabel('特征索引', fontsize=12)
    axes[0].set_ylabel('特征索引', fontsize=12)
    plt.colorbar(im1, ax=axes[0])
    
    # 生成样本相关性
    im2 = axes[1].imshow(synth_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[1].set_title('生成样本特征相关性矩阵', fontsize=14, pad=10)
    axes[1].set_xlabel('特征索引', fontsize=12)
    axes[1].set_ylabel('特征索引', fontsize=12)
    plt.colorbar(im2, ax=axes[1])
    
    plt.suptitle(f'特征相关性矩阵对比（前{n_features}个特征）', 
                 fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ 特征相关性对比图已保存: {output_dir}/correlation_comparison.png")
    plt.close()

def main():
    """主函数：加载训练结果并生成所有可视化"""
    print("=" * 80)
    print("ATAE-GAN Training Results Visualization")
    print("=" * 80)
    
    # 检查文件是否存在
    synthetic_path = Path('results/atae_gan/synthetic.pt')
    if not synthetic_path.exists():
        print(f"Error: Synthetic sample file not found: {synthetic_path}")
        print("Please run run_atae_gan.py first to generate samples")
        return
    
    # 加载生成样本
    print("\nLoading synthetic samples...")
    synthetic_samples = torch.load(synthetic_path)
    print(f"  Synthetic sample shape: {synthetic_samples.shape}")
    
    # 加载训练器（需要从训练脚本中获取）
    print("\nNote: Training loss curves require trainer object")
    print("      If trainer is saved, load it and call visualize_training_history(trainer)")
    
    # 加载真实样本（少数类）用于对比
    print("\nNote: Real samples needed for comparison")
    print("      Please load minority class samples from training data")
    
    print("\n" + "=" * 80)
    print("Visualization Script Ready")
    print("=" * 80)
    print("\nUsage:")
    print("1. Training loss curves: visualize_training_history(trainer)")
    print("2. Sample distribution: visualize_sample_distribution(real_samples, synthetic_samples)")
    print("3. Statistics comparison: visualize_statistics_comparison(real_samples, synthetic_samples)")
    print("4. Correlation comparison: visualize_correlation_comparison(real_samples, synthetic_samples)")

if __name__ == '__main__':
    main()