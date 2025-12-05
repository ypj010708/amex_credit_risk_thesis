import os
import torch
import pandas as pd
from pathlib import Path

from src.training import (
    ATAEGANConfig,
    ATAEGANTrainer,
    preprocess_tabular_dataframe,
)

# -------------------------------
# 1. 载入训练特征与标签
# -------------------------------
train_path = Path('data/raw/train.parquet')
label_path = Path('data/raw/train_labels.csv')  # 若是 parquet 改后缀即可

print('Loading feature table...')
train_df = pd.read_parquet(train_path)

print('Loading labels...')
if label_path.suffix == '.csv':
    label_df = pd.read_csv(label_path)
else:
    label_df = pd.read_parquet(label_path)

if 'customer_ID' not in train_df or 'customer_ID' not in label_df:
    raise ValueError('train/label 文件必须包含 customer_ID 列')

df = train_df.merge(label_df[['customer_ID', 'target']], on='customer_ID', how='left')
df = df.dropna(subset=['target']).reset_index(drop=True)
df['target'] = df['target'].astype(int)

print(f'total rows: {len(df):,}, positives: {df["target"].sum():,}')

# -------------------------------
# 2. 控制数据规模（建议）
# -------------------------------
minority = df[df['target'] == 1]
majority = df[df['target'] == 0]

# 多数类随机抽样，避免占满内存；可按需要调整数量
majority_sample = majority.sample(n=min(len(majority), 200_000), random_state=42)

df_small = pd.concat([minority, majority_sample]).reset_index(drop=True)
print(f'after sampling rows: {len(df_small):,}')

y = df_small['target'].values
X = df_small.drop(columns=['target'])

# -------------------------------
# 3. 自动识别数值/类别列
# -------------------------------
ignore_cols = {'customer_ID'}
numeric_cols = [
    c for c in X.columns
    if pd.api.types.is_numeric_dtype(X[c]) and c not in ignore_cols
]
categorical_cols = [
    c for c in X.columns
    if c not in numeric_cols and c not in ignore_cols
]

print(f'numeric cols: {len(numeric_cols)}, categorical cols: {len(categorical_cols)}')

# -------------------------------
# 4. 预处理成 Tensor
# -------------------------------
tensor, pipeline = preprocess_tabular_dataframe(
    X,
    numeric_cols=numeric_cols,
    categorical_cols=categorical_cols,
    scale_mode='standard',
    use_pca=False,
)

# -------------------------------
# 5. 训练 ATAE-GAN
# -------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = ATAEGANConfig(
    input_dim=tensor.shape[1],
    device=device,
    pretrain_epochs=50,   # 可按资源调节
    adv_epochs=100,
    batch_size=256,
)

trainer = ATAEGANTrainer(cfg)

os.makedirs('results/atae_gan', exist_ok=True)

trainer.fit(
    minority_tensor=tensor[y == 1],
    full_tensor=tensor,
    labels=torch.tensor(y, dtype=torch.float32),
)

# -------------------------------
# 6. 生成合成样本并保存
# -------------------------------
synthetic = trainer.sample(2000)  # 按需调整数量
torch.save(synthetic, 'results/atae_gan/synthetic.pt')
print('Done. Synthetic samples saved to results/atae_gan/synthetic.pt')

# 保存真实少数类样本用于对比
real_minority = tensor[y == 1]
torch.save(real_minority, 'results/atae_gan/real_minority.pt')
print('Real minority samples saved to results/atae_gan/real_minority.pt')

# -------------------------------
# 7. 生成可视化图表
# -------------------------------
print("\n" + "=" * 80)
print("生成可视化图表...")
print("=" * 80)

from visualize_atae_gan import (
    visualize_training_history,
    visualize_sample_distribution,
    visualize_statistics_comparison,
    visualize_correlation_comparison,
)

# 加载真实样本
real_samples = torch.load('results/atae_gan/real_minority.pt')

# 生成所有可视化
visualize_training_history(trainer)
visualize_sample_distribution(real_samples, synthetic)
visualize_statistics_comparison(real_samples, synthetic)
visualize_correlation_comparison(real_samples, synthetic)

print("\n" + "=" * 80)
print("所有可视化图表已生成完成！")
print("保存位置: results/atae_gan/figures/")
print("=" * 80)