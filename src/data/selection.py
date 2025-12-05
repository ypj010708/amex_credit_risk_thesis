"""
级联式特征选择策略
实现论文3.3节描述的特征选择方法
包括：IV（信息价值）、PSI（人口统计学分布稳定性）、Null Importance
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from tqdm.auto import tqdm
import gc
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 3.3.1 基于IV与PSI的特征初筛
# ==========================================

def calculate_iv(df, feature, target, bins=10):
    """
    计算信息价值（Information Value, IV）
    用于衡量特征对目标变量的预测能力
    
    Args:
        df: 数据框
        feature: 特征列名
        target: 目标列名
        bins: 分箱数（对于连续特征）
    Returns:
        IV值
    """
    if feature not in df.columns or target not in df.columns:
        return 0.0
    
    # 处理缺失值
    df_clean = df[[feature, target]].dropna()
    if len(df_clean) == 0:
        return 0.0
    
    # 判断是否为数值型特征
    is_numeric = pd.api.types.is_numeric_dtype(df_clean[feature])
    
    if is_numeric:
        # 对数值型特征进行分箱
        try:
            df_clean['bin'] = pd.cut(df_clean[feature], bins=bins, duplicates='drop')
        except:
            # 如果分箱失败，使用分位数分箱
            df_clean['bin'] = pd.qcut(df_clean[feature], q=bins, duplicates='drop')
    else:
        # 分类特征直接使用原值
        df_clean['bin'] = df_clean[feature]
    
    # 计算每个箱的统计量
    grouped = df_clean.groupby('bin', observed=True).agg({
        target: ['count', 'sum']
    })
    grouped.columns = ['total', 'positive']
    grouped['negative'] = grouped['total'] - grouped['positive']
    
    # 计算总体正负样本数
    total_positive = df_clean[target].sum()
    total_negative = len(df_clean) - total_positive
    
    if total_positive == 0 or total_negative == 0:
        return 0.0
    
    # 计算IV
    iv = 0.0
    for bin_name, row in grouped.iterrows():
        pos_rate = row['positive'] / total_positive if total_positive > 0 else 0
        neg_rate = row['negative'] / total_negative if total_negative > 0 else 0
        
        if pos_rate > 0 and neg_rate > 0:
            woe = np.log(pos_rate / neg_rate)
            iv += (pos_rate - neg_rate) * woe
    
    return iv


def calculate_psi(expected, actual, bins=10):
    """
    计算人口统计学分布稳定性（Population Stability Index, PSI）
    用于衡量特征在训练集和验证集之间分布的变化
    
    Args:
        expected: 基准数据集（训练集）的特征值
        actual: 当前数据集（验证集）的特征值
        bins: 分箱数
    Returns:
        PSI值
    """
    # 处理缺失值
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()
    
    if len(expected) == 0 or len(actual) == 0:
        return np.inf  # 如果数据为空，返回无穷大
    
    # 判断是否为数值型
    is_numeric = pd.api.types.is_numeric_dtype(expected)
    
    if is_numeric:
        # 使用训练集的分位数作为分箱边界
        try:
            bin_edges = pd.qcut(expected, q=bins, duplicates='drop', retbins=True)[1]
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.linspace(expected.min(), expected.max(), bins + 1)
        except:
            bin_edges = np.linspace(expected.min(), expected.max(), bins + 1)
        
        # 计算每个箱的频率
        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)
    else:
        # 分类特征：使用所有唯一值
        all_values = pd.concat([expected, actual]).unique()
        expected_counts = expected.value_counts().reindex(all_values, fill_value=0).values
        actual_counts = actual.value_counts().reindex(all_values, fill_value=0).values
    
    # 转换为频率
    expected_freq = expected_counts / len(expected) if len(expected) > 0 else expected_counts
    actual_freq = actual_counts / len(actual) if len(actual) > 0 else actual_counts
    
    # 避免除零错误
    expected_freq = np.where(expected_freq == 0, 1e-6, expected_freq)
    actual_freq = np.where(actual_freq == 0, 1e-6, actual_freq)
    
    # 计算PSI
    psi = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
    
    return psi


def filter_features_by_iv(X, y, min_iv=0.02, max_iv=None):
    """
    基于IV值进行特征初筛
    
    Args:
        X: 特征数据框
        y: 目标变量
        min_iv: 最小IV阈值（IV < 0.02 的特征通常区分能力较差）
        max_iv: 最大IV阈值（可选，IV过高可能表示过拟合）
    Returns:
        筛选后的特征列表
    """
    print(f"Filtering features by IV (min_iv={min_iv})...")
    
    df = X.copy()
    df['target'] = y
    
    iv_scores = {}
    for col in tqdm(X.columns, desc="Calculating IV"):
        iv = calculate_iv(df, col, 'target')
        iv_scores[col] = iv
    
    # 筛选特征
    selected_features = []
    for col, iv in iv_scores.items():
        if iv >= min_iv:
            if max_iv is None or iv <= max_iv:
                selected_features.append(col)
    
    print(f"IV filtering complete. Selected {len(selected_features)}/{len(X.columns)} features.")
    print(f"IV range: min={min(iv_scores.values()):.4f}, max={max(iv_scores.values()):.4f}")
    
    return selected_features, iv_scores


def filter_features_by_psi(X_train, X_val, max_psi=0.25):
    """
    基于PSI值进行特征稳定性筛选
    
    Args:
        X_train: 训练集特征
        X_val: 验证集特征
        max_psi: 最大PSI阈值（PSI >= 0.25 表示分布变化显著）
    Returns:
        筛选后的特征列表和PSI分数字典
    """
    print(f"Filtering features by PSI (max_psi={max_psi})...")
    
    common_features = set(X_train.columns) & set(X_val.columns)
    psi_scores = {}
    
    for col in tqdm(common_features, desc="Calculating PSI"):
        psi = calculate_psi(X_train[col], X_val[col])
        psi_scores[col] = psi
    
    # 筛选特征
    selected_features = []
    for col, psi in psi_scores.items():
        if psi < max_psi:
            selected_features.append(col)
    
    print(f"PSI filtering complete. Selected {len(selected_features)}/{len(common_features)} features.")
    print(f"PSI range: min={min(psi_scores.values()):.4f}, max={max(psi_scores.values()):.4f}")
    
    return selected_features, psi_scores


def cascade_feature_selection(X_train, y_train, X_val=None, 
                              min_iv=0.02, max_psi=0.25,
                              null_importance=True, null_runs=30):
    """
    级联式特征选择：IV -> PSI -> Null Importance
    
    Args:
        X_train: 训练集特征
        y_train: 训练集标签
        X_val: 验证集特征（可选，用于PSI计算）
        min_iv: IV最小阈值
        max_psi: PSI最大阈值
        null_importance: 是否使用Null Importance进行精选
        null_runs: Null Importance运行次数
    Returns:
        最终选中的特征列表
    """
    print("=" * 60)
    print("Starting Cascade Feature Selection")
    print("=" * 60)
    
    # 第一步：IV初筛
    print("\nStep 1: IV-based initial screening...")
    iv_selected, iv_scores = filter_features_by_iv(X_train, y_train, min_iv=min_iv)
    X_train_iv = X_train[iv_selected]
    
    # 第二步：PSI稳定性筛选（如果有验证集）
    if X_val is not None:
        print("\nStep 2: PSI-based stability screening...")
        X_val_iv = X_val[iv_selected]
        psi_selected, psi_scores = filter_features_by_psi(X_train_iv, X_val_iv, max_psi=max_psi)
        X_train_psi = X_train_iv[psi_selected]
    else:
        print("\nStep 2: Skipping PSI (no validation set provided)")
        psi_selected = iv_selected
        X_train_psi = X_train_iv
    
    # 第三步：Null Importance精选
    if null_importance:
        print("\nStep 3: Null Importance-based fine selection...")
        final_selected = run_null_importance_selection(
            X_train_psi, y_train, n_runs=null_runs
        )
    else:
        print("\nStep 3: Skipping Null Importance")
        final_selected = psi_selected
    
    print("=" * 60)
    print(f"Feature selection complete. Final: {len(final_selected)}/{len(X_train.columns)} features")
    print("=" * 60)
    
    return final_selected

# ==========================================
# 3.3.2 基于Null Importance的特征精选
# ==========================================

def run_null_importance_selection(X, y, n_runs=30, importance_type='gain', keep_threshold_percentile=75, random_state=42):
    """
    使用 Null Importance 策略进行特征精选。
    通过多次打乱标签训练模型，建立特征重要性的噪声分布，仅保留真实重要性显著高于噪声分布的特征。
    
    Args:
        X (pd.DataFrame): 特征矩阵.
        y (pd.Series): 标签.
        n_runs (int): 打乱标签运行的次数 (建议 30-50 次).
        importance_type (str): LGBM 重要性类型 ('split' 或 'gain'). 'gain' 通常更准确.
        keep_threshold_percentile (int): 保留阈值的分位数 (例如 75 表示保留真实重要性 > 75分位Null重要性的特征).
        
    Returns:
        list: 被选中的特征名列表.
    """
    print(f"Starting Null Importance Selection with {n_runs} runs...")
    features = X.columns.tolist()
    
    lgb_params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': 150,       # 不需要太多树，快速拟合即可
        'learning_rate': 0.1,
        'num_leaves': 31,
        'random_state': random_state,
        'n_jobs': -1,
        'importance_type': importance_type,
        'verbose': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    # 1. 计算原始标签的特征重要性 (Actual Importance)
    print("Calculating actual feature importance (Ground Truth)...")
    clf_actual = lgb.LGBMClassifier(**lgb_params)
    clf_actual.fit(X, y)
    
    actual_imp_df = pd.DataFrame({
        'feature': features,
        'actual_importance': clf_actual.feature_importances_
    })
    
    # 2. 计算 Null Importance (多次打乱标签)
    print(f"Running {n_runs} permutations to Generate Null Distributions...")
    null_importances = []
    for i in tqdm(range(n_runs)):
        # 核心：随机打乱标签，打破特征与标签的真实关联
        y_shuffled = np.random.permutation(y)
        
        # 使用不同的随机种子
        params_null = lgb_params.copy()
        params_null['random_state'] = random_state + i + 1
        
        clf_null = lgb.LGBMClassifier(**params_null)
        clf_null.fit(X, y_shuffled)
        
        null_importances.append(clf_null.feature_importances_)
        gc.collect()
        
    # 构建 Null Importance 矩阵 (行: 特征, 列: 每次运行)
    null_imp_matrix = np.array(null_importances).T
    
    # 3. 比较与筛选
    print("Calculating thresholds and selecting features...")
    selected_features = []
    
    for idx, feature in enumerate(features):
        actual_val = actual_imp_df.loc[idx, 'actual_importance']
        null_vals = null_imp_matrix[idx, :]
        
        # 计算该特征 Null 分布指定的百分位数作为阈值
        null_threshold = np.percentile(null_vals, keep_threshold_percentile)
        
        # 筛选条件：真实重要性必须大于噪声阈值
        # 可以增加额外条件：真实重要性本身不能太接近0
        if actual_val > null_threshold and actual_val > 1e-3:
            selected_features.append(feature)
            
    print(f"Done. Selected {len(selected_features)} features out of {len(features)}.")
    print(f"Dropped {len(features) - len(selected_features)} noise/redundant features.")
    
    return selected_features
