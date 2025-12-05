import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import warnings
import os
import sys
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.csv as pacsv
except ImportError:
    pa = None
    pacsv = None

# --- 设置编码以支持中文输出 ---
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# --- 配置绘图样式和忽略警告 ---
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')

# --- 确保图片保存目录存在 ---
os.makedirs('results/figures', exist_ok=True)

# ==============================================================================
# 1. 加载真实数据集 (Real Data Loading)
# ==============================================================================

def load_amex_data(train_path='data/raw/train.parquet', sample_size=None):
    """
    加载 Amex 数据集
    参数:
        train_path: 训练数据路径
        sample_size: 如果指定，则只加载部分数据用于快速可视化
    """
    print("正在加载 Amex 数据集...")
    try:
        # 检查可用的parquet引擎
        has_pyarrow = False
        has_fastparquet = False
        try:
            import pyarrow
            has_pyarrow = True
        except:
            pass
        try:
            import fastparquet
            has_fastparquet = True
        except:
            pass
        
        if not has_pyarrow and not has_fastparquet:
            raise ImportError(
                "缺少parquet读取库！请安装以下任一库：\n"
                "  - pip install pyarrow  (推荐)\n"
                "  - pip install fastparquet\n"
                "  - conda install pyarrow -c conda-forge"
            )
        
        # 尝试使用 pyarrow，如果失败则使用 fastparquet
        if has_pyarrow:
            try:
                df = pd.read_parquet(train_path, engine='pyarrow')
            except Exception as e:
                if has_fastparquet:
                    print(f"  警告: pyarrow读取失败，尝试使用fastparquet: {e}")
                    df = pd.read_parquet(train_path, engine='fastparquet')
                else:
                    raise
        else:
            df = pd.read_parquet(train_path, engine='fastparquet')
        
        if sample_size is not None and len(df) > sample_size:
            # 如果是时序数据（有customer_ID），按客户采样；否则按target采样
            if 'customer_ID' in df.columns:
                # 按客户采样，保持类别比例
                if 'target' in df.columns:
                    # 先获取每个客户的target（同一客户的所有记录target相同）
                    customer_targets = df.groupby('customer_ID')['target'].first()
                    # 估算需要的客户数量（假设每个客户平均10-13条记录）
                    avg_records_per_customer = len(df) / len(customer_targets)
                    n_customers_needed = int(sample_size / avg_records_per_customer)
                    # 按target分组采样客户
                    sampled_customers = customer_targets.groupby(customer_targets, group_keys=False).apply(
                        lambda x: x.sample(min(len(x), n_customers_needed // 2), random_state=42)
                    ).index
                    df = df[df['customer_ID'].isin(sampled_customers)].copy()
                else:
                    # 没有target，随机采样客户
                    unique_customers = df['customer_ID'].unique()
                    avg_records = len(df) / len(unique_customers)
                    n_customers = min(len(unique_customers), int(sample_size / avg_records))
                    sampled_customers = np.random.choice(unique_customers, n_customers, replace=False)
                    df = df[df['customer_ID'].isin(sampled_customers)].copy()
            elif 'target' in df.columns:
                # 非时序数据，按target采样
                df_sampled = df.groupby('target', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), sample_size // 2), random_state=42)
                ).reset_index(drop=True)
                df = df_sampled
            else:
                df = df.sample(n=sample_size, random_state=42)
        
        print(f"Amex 数据加载完成: {len(df)} 条记录, {len(df.columns)} 个特征")
        if 'target' in df.columns:
            print(f"Target 分布: {df['target'].value_counts().to_dict()}")
        return df
    except Exception as e:
        print(f"加载 Amex 数据时出错: {e}")
        return None

def load_lendingclub_data(accepted_path='data/raw/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv', 
                          sample_size=None):
    """
    加载 LendingClub 数据集并构建目标变量
    参数:
        accepted_path: accepted 数据路径
        sample_size: 如果指定，则只加载部分数据用于快速可视化
    """
    print("正在加载 LendingClub 数据集...")
    try:
        # 需要的列（覆盖可视化所需字段）
        lc_columns = [
            'loan_amnt', 'annual_inc', 'dti', 'fico_range_low', 'fico_range_high',
            'issue_d', 'loan_status', 'int_rate', 'term', 'emp_length'
        ]

        # 优先使用 PyArrow 加载，可显著降低内存占用
        if pacsv is not None:
            column_types = {
                'loan_amnt': pa.float32(),
                'annual_inc': pa.float32(),
                'dti': pa.float32(),
                'fico_range_low': pa.float32(),
                'fico_range_high': pa.float32(),
                'int_rate': pa.float32(),
                'term': pa.string(),
                'emp_length': pa.string(),
                'issue_d': pa.string(),
                'loan_status': pa.string(),
            }
            read_opts = pacsv.ReadOptions(use_threads=True, block_size=1 << 22)
            convert_opts = pacsv.ConvertOptions(
                include_columns=[col for col in lc_columns],
                column_types=column_types,
                strings_can_be_null=True
            )
            arrow_table = pacsv.read_csv(
                accepted_path,
                read_options=read_opts,
                convert_options=convert_opts
            )
            df = arrow_table.to_pandas(types_mapper=pd.ArrowDtype, self_destruct=True).copy()
            for col in ['fico_range_low', 'fico_range_high']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')
        else:
            df = pd.read_csv(
                accepted_path,
                usecols=lc_columns,
                dtype={
                    'loan_amnt': 'float32',
                    'annual_inc': 'float32',
                    'dti': 'float32',
                    'fico_range_low': 'float32',
                    'fico_range_high': 'float32',
                    'int_rate': 'float32'
                },
                low_memory=False
            )
        
        # 筛选2015-2018年的数据
        if 'issue_d' in df.columns:
            df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
            df = df[(df['issue_d'].dt.year >= 2015) & (df['issue_d'].dt.year <= 2018)]
        
        # 构建目标变量
        # "Fully Paid" -> 0 (正常), "Charged Off" -> 1 (违约)
        if 'loan_status' in df.columns:
            df['target'] = df['loan_status'].map({
                'Fully Paid': 0,
                'Charged Off': 1
            })
            # 只保留有明确标签的样本，剔除中间状态
            df = df[df['target'].notna()].copy()
        
        if sample_size is not None and len(df) > sample_size:
            # 保持类别比例进行采样
            if 'target' in df.columns:
                df_sampled = df.groupby('target', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), sample_size // 2))
                ).reset_index(drop=True)
                df = df_sampled
            else:
                df = df.sample(n=sample_size, random_state=42)
        
        print(f"LendingClub 数据加载完成: {len(df)} 条记录, {len(df.columns)} 个特征")
        if 'target' in df.columns:
            print(f"Target 分布: {df['target'].value_counts().to_dict()}")
        return df
    except Exception as e:
        print(f"加载 LendingClub 数据时出错: {e}")
        return None

# 加载数据
print("=" * 60)
print("开始加载数据集...")
print("=" * 60)

# ==============================================================================
# 数据加载配置
# 如果数据量很大，可以设置 sample_size 来加快可视化速度
# 设置为 None 则加载全部数据
# ==============================================================================
USE_SAMPLING = False  # 设置为 False 以加载全部数据
AMEX_SAMPLE_SIZE = 50000 if USE_SAMPLING else None  # Amex采样5万条记录（或全部）
LC_SAMPLE_SIZE = 100000 if USE_SAMPLING else None  # LendingClub采样10万条记录（或全部）

if USE_SAMPLING:
    print("注意: 当前使用采样模式以加快可视化速度")
    print("如需加载全部数据，请将 USE_SAMPLING 设置为 False")
else:
    print("注意: 当前加载全部数据，可能需要较长时间")

df_amex = load_amex_data(sample_size=AMEX_SAMPLE_SIZE)
df_lc = load_lendingclub_data(sample_size=LC_SAMPLE_SIZE)

# 即使部分数据加载失败，也继续运行（只生成可用的可视化）
if df_lc is None:
    print("错误: LendingClub 数据加载失败，无法继续")
    exit(1)

if df_amex is None:
    print("警告: Amex 数据加载失败，将只生成 LendingClub 的可视化")
    print("提示: 请安装 pyarrow 或 fastparquet: pip install pyarrow")

# ==============================================================================
# 2. 可视化分析 (Visualization Analysis)
# ==============================================================================

# --- 3.1.2 数据分布特征、缺失机制与异常值分析 ---

print("\n" + "=" * 60)
print("--- 开始进行 3.1.2 节的可视化分析 ---")
print("=" * 60)

# 1. 数据分布特征：非正态性与长尾分布
print("\n1. 绘制数据分布特征图...")

# 找到合适的特征列
# LendingClub: annual_inc (年收入)
# Amex: 找到以S_开头的消费特征或数值特征
lc_income_col = 'annual_inc' if 'annual_inc' in df_lc.columns else None
amex_spend_col = None
if df_amex is not None:
    amex_spend_cols = [col for col in df_amex.columns if col.startswith('S_') and df_amex[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    amex_spend_col = amex_spend_cols[0] if amex_spend_cols else None

if lc_income_col:
    if amex_spend_col and df_amex is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    
    # LendingClub 年收入分布
    lc_data = df_lc[lc_income_col].dropna()
    if len(lc_data) > 0:
        # 处理异常值，只显示合理范围
        q99 = lc_data.quantile(0.99)
        lc_data_plot = lc_data[lc_data <= q99]
        sns.histplot(lc_data_plot, kde=True, ax=axes[0], bins=50)
        axes[0].set_title('LendingClub: 年收入 (annual_inc) 分布 (长尾)', fontsize=12)
        axes[0].set_xlabel('年收入 (美元)')
        axes[0].set_ylabel('频数')
    
    # Amex 消费特征分布
    if amex_spend_col and df_amex is not None:
        amex_data = df_amex[amex_spend_col].dropna()
        if len(amex_data) > 0:
            # 处理异常值
            q99 = amex_data.quantile(0.99)
            amex_data_plot = amex_data[amex_data <= q99]
            sns.histplot(amex_data_plot, kde=True, ax=axes[1], bins=50, color='orange')
            axes[1].set_title(f'Amex: 消费特征 ({amex_spend_col}) 分布 (长尾)', fontsize=12)
            axes[1].set_xlabel('消费特征值')
            axes[1].set_ylabel('频数')
    
    plt.suptitle('图 3-1: 数据分布特征分析 - 长尾分布', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/figures/fig_3_1_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print("   ✓ 图 3-1 已保存")
else:
    print("   ⚠ 未找到合适的特征列，跳过分布图")

# 2. 缺失机制分析
print("\n2. 绘制缺失机制分析图...")

def plot_missing_overview(df, dataset_name, axes):
    """绘制缺失率条形图 + 缺失共现热力图"""
    missing_pct = df.isnull().mean().sort_values(ascending=False)
    if missing_pct.empty:
        axes[0].text(0.5, 0.5, f"{dataset_name}\n无缺失值", ha='center', va='center')
        axes[1].axis('off')
        return
    
    top_cols = missing_pct.head(15)
    sns.barplot(x=top_cols.values * 100, y=top_cols.index, ax=axes[0], palette='viridis')
    axes[0].set_xlabel('缺失率 (%)')
    axes[0].set_ylabel('特征')
    axes[0].set_title(f'{dataset_name}: Top15 特征缺失率')
    axes[0].grid(axis='x', linestyle='--', alpha=0.3)
    
    # 缺失共现：计算缺失指示矩阵的相关性
    miss_matrix = df[top_cols.index].isnull().astype(int)
    if miss_matrix.shape[1] >= 2:
        corr = miss_matrix.corr()
        sns.heatmap(corr, ax=axes[1], cmap='Reds', vmin=0, vmax=1, cbar_kws={'label': '缺失共现系数'})
        axes[1].set_title(f'{dataset_name}: 缺失共现热力图')
    else:
        axes[1].text(0.5, 0.5, '特征数量不足，无法计算共现', ha='center', va='center')
        axes[1].axis('off')

# 准备采样数据以保证可视化效率
lc_missing_df = df_lc.sample(min(20000, len(df_lc)), random_state=42)
amex_missing_df = None
if df_amex is not None:
    if 'customer_ID' in df_amex.columns:
        sampled_ids = df_amex['customer_ID'].drop_duplicates().sample(
            min(2000, df_amex['customer_ID'].nunique()), random_state=42)
        amex_missing_df = df_amex[df_amex['customer_ID'].isin(sampled_ids)].copy()
    else:
        amex_missing_df = df_amex.sample(min(20000, len(df_amex)), random_state=42).copy()

if lc_missing_df is not None:
    if amex_missing_df is not None:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        plot_missing_overview(lc_missing_df, 'LendingClub', axes[0])
        plot_missing_overview(amex_missing_df, 'Amex', axes[1])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        plot_missing_overview(lc_missing_df, 'LendingClub', axes)
    plt.suptitle('图 3-2: 缺失机制分析（缺失率 + 共现关系）', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/figures/fig_3_2_missing_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ 图 3-2 已保存")
else:
    print("   ⚠ LendingClub 数据缺失率可视化失败，跳过图 3-2")

# 3. 异常值分析：箱形图 + 极端点
print("\n3. 绘制异常值分析图...")

def prepare_outlier_series(series, clip_quantile=0.999, log_transform=False):
    """根据配置对长尾数据做预处理，减少极端值占据视觉"""
    data = series.dropna().astype('float32')
    if data.empty:
        return data
    if log_transform:
        data = np.log1p(np.clip(data, a_min=0, a_max=None))
    else:
        upper = data.quantile(clip_quantile)
        data = data[data <= upper]
    return data


def plot_outliers(ax, data, title, xlabel, outlier_sample_ratio=1.0):
    """使用箱线 + 散点展示异常值位置"""
    if data.empty:
        ax.text(0.5, 0.5, '无数据', ha='center', va='center')
        ax.axis('off')
        return
    sns.boxplot(x=data, ax=ax, color='#7FB3D5', whis=1.5, fliersize=0)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1 if q3 > q1 else 1e-6
    upper = q3 + 1.5 * iqr
    outliers = data[data > upper]
    if not outliers.empty:
        if outlier_sample_ratio < 1.0:
            sample_size = max(1, int(len(outliers) * outlier_sample_ratio))
            outliers = outliers.sample(sample_size, random_state=42)
        jitter = np.random.uniform(-0.02, 0.02, size=len(outliers))
        ax.scatter(outliers, jitter, color='#E74C3C', alpha=0.5, s=18, label='异常点 (>Q3+1.5IQR)')
        ax.legend(loc='lower right', fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

lc_loan_col = 'loan_amnt' if 'loan_amnt' in df_lc.columns else None

if lc_loan_col:
    if amex_spend_col and df_amex is not None:
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(9, 5))
        axes = [axes]
    
    lc_sample_raw = df_lc[lc_loan_col].sample(
        min(50000, df_lc[lc_loan_col].notna().sum()), random_state=42)
    lc_sample_for_outlier = prepare_outlier_series(lc_sample_raw, clip_quantile=0.995)
    plot_outliers(axes[0], lc_sample_for_outlier, 'LendingClub: 贷款金额异常值', '贷款金额 (美元)')
    
    if amex_spend_col and df_amex is not None:
        amex_sample_raw = df_amex[amex_spend_col].sample(
            min(80000, df_amex[amex_spend_col].notna().sum()), random_state=42)
        amex_sample_for_outlier = prepare_outlier_series(
            amex_sample_raw, clip_quantile=0.999, log_transform=True)
        plot_outliers(
            axes[1],
            amex_sample_for_outlier,
            f'Amex: 消费特征 ({amex_spend_col}) 异常值（log1p）',
            'log1p(消费特征值)',
            outlier_sample_ratio=0.01
        )
    
    plt.suptitle('图 3-3: 异常值分析（长尾平滑处理）', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/figures/fig_3_3_outliers_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ 图 3-3 已保存")
else:
    print("   ⚠ 未找到合适的特征列，跳过异常值分析图")

# --- 3.1.3 样本不平衡程度与时序结构分析 ---

print("\n" + "=" * 60)
print("--- 开始进行 3.1.3 节的可视化分析 ---")
print("=" * 60)

# 4. 样本不平衡程度
print("\n4. 绘制样本不平衡程度分析图...")

if 'target' in df_lc.columns:
    if df_amex is not None and 'target' in df_amex.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(7, 6))
        axes = [axes]
    
    # LendingClub 样本不平衡
    lc_counts = df_lc['target'].value_counts().sort_index()
    lc_ratio = lc_counts[1] / lc_counts[0] if len(lc_counts) > 1 and lc_counts[0] > 0 else 0
    axes[0].pie(lc_counts, labels=[f'正常 (0): {lc_counts[0]:,}', f'违约 (1): {lc_counts[1]:,}'], 
                autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
    axes[0].set_title(f'LendingClub: 样本不平衡程度 (约 1:{lc_ratio:.1f})', fontsize=12)
    
    # Amex 样本不平衡（需要按客户去重）
    if df_amex is not None and 'target' in df_amex.columns:
        if 'customer_ID' in df_amex.columns:
            amex_unique = df_amex.drop_duplicates('customer_ID')
            amex_counts = amex_unique['target'].value_counts().sort_index()
        else:
            amex_counts = df_amex['target'].value_counts().sort_index()
        
        amex_ratio = amex_counts[1] / amex_counts[0] if len(amex_counts) > 1 and amex_counts[0] > 0 else 0
        axes[1].pie(amex_counts, labels=[f'正常 (0): {amex_counts[0]:,}', f'违约 (1): {amex_counts[1]:,}'], 
                    autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
        axes[1].set_title(f'Amex: 样本不平衡程度 (约 1:{amex_ratio:.1f})', fontsize=12)
    
    plt.suptitle('图 3-4: 样本不平衡程度分析', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('results/figures/fig_3_4_class_imbalance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✓ 图 3-4 已保存")
else:
    print("   ⚠ 未找到 target 列，跳过样本不平衡分析图")

# 5. 时序结构分析：违约与正常用户行为模式对比
print("\n5. 绘制时序结构分析图...")

if df_amex is not None and 'customer_ID' in df_amex.columns and 'target' in df_amex.columns:
    # 找到时间列（通常是 S_2 或其他日期列）
    time_col = None
    for col in ['S_2', 'month', 'date']:
        if col in df_amex.columns:
            time_col = col
            break
    
    # 找到余额相关特征（B_ 开头）
    balance_cols = [col for col in df_amex.columns if col.startswith('B_') and df_amex[col].dtype in ['float64', 'float32', 'int64', 'int32']]
    balance_col = balance_cols[0] if balance_cols else None
    
    if balance_col:
        # 如果有时间列，按时间分组；否则按 customer_ID 的序号分组
        if time_col:
            # 转换时间列
            if df_amex[time_col].dtype == 'object':
                df_amex[time_col] = pd.to_datetime(df_amex[time_col], errors='coerce')
            # 创建月份序号
            df_amex_sorted = df_amex.sort_values(['customer_ID', time_col])
            df_amex_sorted['month_seq'] = df_amex_sorted.groupby('customer_ID').cumcount()
        else:
            # 如果没有明确的时间列，假设每条记录代表一个月
            df_amex_sorted = df_amex.copy()
            df_amex_sorted['month_seq'] = df_amex_sorted.groupby('customer_ID').cumcount()
        
        # 计算每个月份序号的均值（按target分组）
        amex_ts_analysis = df_amex_sorted.groupby(['month_seq', 'target'])[balance_col].mean().reset_index()
        
        plt.figure(figsize=(12, 7))
        sns.lineplot(data=amex_ts_analysis, x='month_seq', y=balance_col, hue='target', 
                     style='target', markers=True, dashes=False, palette={0: 'blue', 1: 'red'}, linewidth=2)
        plt.title(f'图 3-5: Amex 时序结构分析 - 不同类别用户平均余额 ({balance_col}) 变化趋势', fontsize=14)
        plt.xlabel('月份序号 (Month Sequence)')
        plt.ylabel(f'平均余额特征值 ({balance_col})')
        plt.legend(title='用户类别', labels=['正常用户', '违约用户'], loc='best')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.savefig('results/figures/fig_3_5_time_series_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✓ 图 3-5 已保存")
    else:
        print("   ⚠ 未找到余额特征列，跳过时序结构分析图")
else:
    print("   ⚠ Amex 数据缺少 customer_ID 或 target 列，跳过时序结构分析图")

print("\n" + "=" * 60)
print("所有可视化分析完成！图片已保存到 results/figures/ 目录")
print("=" * 60)
