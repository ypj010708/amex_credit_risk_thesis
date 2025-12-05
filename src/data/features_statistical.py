"""
统计视图构建模块
实现论文3.2.2节描述的时窗聚合特征和趋势斜率特征
"""
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# 延迟导入sklearn
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # 简单的线性回归实现
    class LinearRegression:
        def __init__(self):
            self.coef_ = None
        def fit(self, X, y):
            # 简单的最小二乘法
            X = np.array(X).reshape(-1, 1) if len(X.shape) == 1 else np.array(X)
            y = np.array(y).reshape(-1, 1) if len(y.shape) == 1 else np.array(y)
            X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])
            coef = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            self.coef_ = coef[1:]  # 去掉截距
            return self

# ==========================================
# 核心创新点实现：趋势特征计算
# ==========================================

def _calculate_slope(series):
    """辅助函数：计算单个序列的线性回归斜率 (针对已排序的时间序列)"""
    # 移除 NaN 值
    series = series.dropna()
    if len(series) < 2:
        return 0.0 # 数据点不足，认为无趋势或斜率为0
    
    # 使用索引作为 X 轴 (代表相对时间步)
    X = np.arange(len(series))
    y = series.values
    
    if SKLEARN_AVAILABLE:
        # 使用sklearn
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), y.reshape(-1, 1))
        return model.coef_[0][0] if hasattr(model.coef_[0], '__len__') else model.coef_[0]
    else:
        # 手动计算斜率（最小二乘法）
        n = len(X)
        sum_x = X.sum()
        sum_y = y.sum()
        sum_xy = (X * y).sum()
        sum_x2 = (X * X).sum()
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0.0
        return slope

def build_trend_features(df, target_features, id_col='customer_ID', 
                        time_col='S_2', time_windows=[3, 6]):
    """
    构建趋势斜率特征
    计算指定特征在最近 N 个月内的变化斜率，如负债率恶化趋势、还款能力恶化趋势等
    
    Args:
        df: 原始交易数据，必须包含ID列和时间列
        target_features: 需要计算趋势的关键数值特征名，如 ['P_2', 'B_1', 'D_39', 'dti']
        id_col: ID列名
        time_col: 时间列名
        time_windows: 时间窗口列表（月数）
    Returns:
        趋势特征数据框（每个ID一行）
    """
    print(f"Building trend slope features for {target_features} over windows {time_windows}...")
    
    df = df.copy()
    
    # 确保时间列为日期格式并按时间排序
    if time_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values([id_col, time_col])
    else:
        # 如果没有时间列，按索引顺序排序
        df = df.sort_values([id_col])
    
    trend_data = []
    unique_ids = df[id_col].unique()
    
    for uid in tqdm(unique_ids, desc="Calculating Trends"):
        group = df[df[id_col] == uid].copy()
        if len(group) < 2:
            # 数据点不足，跳过
            continue
        
        trend_dict = {id_col: uid}
        
        # 获取最新时间点
        if time_col in group.columns:
            latest_date = group[time_col].max()
        else:
            latest_date = None
        
        for feature in target_features:
            if feature not in group.columns:
                continue
            
            # 计算全局趋势（如果有时间列）
            if time_col in group.columns and latest_date is not None:
                for window in time_windows:
                    start_date = latest_date - pd.Timedelta(days=window * 30)
                    window_data = group[group[time_col] >= start_date].sort_values(time_col)
                    
                    if len(window_data) >= 2:
                        slope = _calculate_slope(window_data[feature])
                        trend_dict[f'{feature}_trend_{window}m_slope'] = slope
                    else:
                        trend_dict[f'{feature}_trend_{window}m_slope'] = 0.0
            else:
                # 如果没有时间列，使用所有数据计算趋势
                for window in time_windows:
                    window_data = group[feature].tail(window)
                    if len(window_data) >= 2:
                        slope = _calculate_slope(window_data)
                        trend_dict[f'{feature}_trend_{window}m_slope'] = slope
                    else:
                        trend_dict[f'{feature}_trend_{window}m_slope'] = 0.0
        
        trend_data.append(trend_dict)
    
    df_trend = pd.DataFrame(trend_data)
    # 对计算产生的缺失值填0 (表示无趋势)
    df_trend = df_trend.fillna(0.0)
    print(f"Trend features complete. Shape: {df_trend.shape}")
    return df_trend

# ==========================================
# 标准统计聚合
# ==========================================
def build_agg_features(df, num_features, cat_features, id_col='customer_ID'):
    """
    构建基础统计聚合特征 (Mean, Std, Min, Max, Last)
    
    Args:
        df: 输入数据框
        num_features: 数值型特征列表
        cat_features: 分类型特征列表
        id_col: ID列名
    Returns:
        聚合后的特征数据框
    """
    print("Building standard aggregation features...")
    
    df = df.copy()
    
    # 确保按时间排序以便获取 'last'（如果有时间列）
    time_col = None
    for col in ['S_2', 'issue_d', 'date']:
        if col in df.columns:
            time_col = col
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.sort_values([id_col, time_col])
            break
    
    if time_col is None:
        df = df.sort_values([id_col])
    
    agg_funcs = {}
    for col in num_features:
        if col in df.columns:
            agg_funcs[col] = ['mean', 'std', 'min', 'max', 'last', 'median']
    
    for col in cat_features:
        if col in df.columns:
            # 对于分类特征，计算唯一值数量和最后一个状态
            agg_funcs[col] = ['nunique', 'last']
    
    if len(agg_funcs) == 0:
        print("Warning: No valid features for aggregation")
        return pd.DataFrame({id_col: df[id_col].unique()})
    
    df_agg = df.groupby(id_col).agg(agg_funcs)
    
    # 展平多级列名 (例如: P_2_mean, P_2_std)
    df_agg.columns = [f"{col}_{stat}" for col, stat in df_agg.columns]
    df_agg = df_agg.reset_index()
    
    print(f"Aggregation complete. Shape: {df_agg.shape}")
    return df_agg

# ==========================================
# 时窗聚合特征（论文3.2.2节）
# ==========================================

def build_time_window_features(df, id_col='customer_ID', time_col='S_2',
                               value_cols=None, windows=[3, 6]):
    """
    构建时窗聚合特征
    例如：近3个月平均还款额、近6个月消费金额平均值等
    
    Args:
        df: 输入数据框（时序数据）
        id_col: ID列名
        time_col: 时间列名
        value_cols: 需要聚合的数值特征列表
        windows: 时间窗口列表（月数）
    Returns:
        聚合后的特征数据框（每个ID一行）
    """
    print(f"Building time window aggregation features for windows {windows}...")
    
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if id_col in value_cols:
            value_cols.remove(id_col)
        if time_col in value_cols:
            value_cols.remove(time_col)
    
    df = df.copy()
    
    # 确保时间列为datetime类型
    if time_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values([id_col, time_col])
    
    agg_data = []
    unique_ids = df[id_col].unique()
    
    for uid in tqdm(unique_ids, desc="Time Window Aggregation"):
        group = df[df[id_col] == uid].copy()
        if len(group) == 0:
            continue
        
        agg_dict = {id_col: uid}
        
        # 获取最新时间点
        if time_col in group.columns:
            latest_time = group[time_col].max()
        else:
            latest_time = None
        
        for col in value_cols:
            if col not in group.columns:
                continue

            if latest_time is not None:
                for window in windows:
                    window_start = latest_time - pd.Timedelta(days=window * 30)
                    window_data = group[group[time_col] >= window_start][col]
                    
                    if len(window_data) > 0:
                        agg_dict[f'{col}_avg_last_{window}m'] = window_data.mean()
                        agg_dict[f'{col}_std_last_{window}m'] = window_data.std() if len(window_data) > 1 else 0.0
                        agg_dict[f'{col}_min_last_{window}m'] = window_data.min()
                        agg_dict[f'{col}_max_last_{window}m'] = window_data.max()
                    else:
                        agg_dict[f'{col}_avg_last_{window}m'] = 0.0
                        agg_dict[f'{col}_std_last_{window}m'] = 0.0
                        agg_dict[f'{col}_min_last_{window}m'] = 0.0
                        agg_dict[f'{col}_max_last_{window}m'] = 0.0
            else:
                # 如果没有时间列，使用最近N条记录
                for window in windows:
                    window_data = group[col].tail(window)
                    if len(window_data) > 0:
                        agg_dict[f'{col}_avg_last_{window}m'] = window_data.mean()
                        agg_dict[f'{col}_std_last_{window}m'] = window_data.std() if len(window_data) > 1 else 0.0
                    else:
                        agg_dict[f'{col}_avg_last_{window}m'] = 0.0
                        agg_dict[f'{col}_std_last_{window}m'] = 0.0
        
        agg_data.append(agg_dict)
    
    df_agg = pd.DataFrame(agg_data)
    df_agg = df_agg.fillna(0.0)
    print(f"Time window aggregation complete. Shape: {df_agg.shape}")
    return df_agg


def build_lendingclub_features(df):
    """
    为LendingClub数据集构建特定的信贷特征
    根据论文描述，构建如近3个月平均还款额等特征
    
    Args:
        df: LendingClub数据框
    Returns:
        添加了特征的数据框
    """
    print("Building LendingClub-specific credit features...")
    df = df.copy()
    
    # 计算债务收入比相关特征
    if 'dti' in df.columns:
        df['dti_category'] = pd.cut(df['dti'], 
                                    bins=[0, 10, 20, 30, 50, 100],
                                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # 计算年收入相关特征
    if 'annual_inc' in df.columns:
        df['annual_inc_log'] = np.log1p(df['annual_inc'])
        df['annual_inc_category'] = pd.qcut(df['annual_inc'], 
                                           q=5, 
                                           labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                           duplicates='drop')
    
    # 计算贷款金额相关特征
    if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
        df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    
    # FICO分数相关特征
    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
        df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        df['fico_range'] = df['fico_range_high'] - df['fico_range_low']
        df['fico_category'] = pd.cut(df['fico_score'],
                                    bins=[0, 580, 670, 740, 800, 850],
                                    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
    
    # 利率相关特征
    if 'int_rate' in df.columns:
        df['int_rate_category'] = pd.cut(df['int_rate'],
                                        bins=[0, 5, 10, 15, 20, 30],
                                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    return df

# ==========================================
# 趋势斜率特征（论文3.2.2节）
# ==========================================
