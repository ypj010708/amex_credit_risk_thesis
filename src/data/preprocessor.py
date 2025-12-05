"""
数据预处理与清洗模块
实现论文3.2.1节描述的数据预处理策略
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 延迟导入scipy以避免问题
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # 定义占位函数
    class stats:
        @staticmethod
        def zscore(x):
            return np.abs((x - np.mean(x)) / np.std(x))

# 延迟导入sklearn以避免pyarrow DLL问题
try:
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: sklearn not available: {e}")
    SKLEARN_AVAILABLE = False
    # 定义占位类
    class KNNImputer:
        def __init__(self, *args, **kwargs):
            raise ImportError("sklearn not available")
    class SimpleImputer:
        def __init__(self, *args, **kwargs):
            raise ImportError("sklearn not available")
    class StandardScaler:
        def __init__(self, *args, **kwargs):
            raise ImportError("sklearn not available")
    class MinMaxScaler:
        def __init__(self, *args, **kwargs):
            raise ImportError("sklearn not available")


class DataPreprocessor:
    """
    数据预处理类，实现缺失值处理、异常值检测、标准化等功能
    """
    
    def __init__(self, dataset_type='amex'):
        """
        Args:
            dataset_type: 'amex' 或 'lendingclub'
        """
        self.dataset_type = dataset_type
        self.scalers = {}
        self.imputers = {}
        self.feature_stats = {}
        
    def remove_duplicates(self, df, id_col='customer_ID', time_col=None):
        """
        删除重复数据
        Args:
            df: 输入数据框
            id_col: ID列名
            time_col: 时间列名（如果有）
        Returns:
            去重后的数据框
        """
        print("Removing duplicates...")
        initial_len = len(df)
        
        if time_col and time_col in df.columns:
            # 基于ID和时间戳去重
            df = df.drop_duplicates(subset=[id_col, time_col], keep='first')
        else:
            # 基于ID去重
            if id_col in df.columns:
                df = df.drop_duplicates(subset=[id_col], keep='first')
            else:
                df = df.drop_duplicates(keep='first')
        
        removed = initial_len - len(df)
        print(f"Removed {removed} duplicate records ({removed/initial_len*100:.2f}%)")
        return df.reset_index(drop=True)
    
    def detect_outliers_iqr(self, series, factor=1.5):
        """
        使用IQR方法检测异常值
        Args:
            series: 数据序列
            factor: IQR倍数因子
        Returns:
            异常值索引的布尔数组
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def detect_outliers_zscore(self, series, threshold=3):
        """
        使用Z-score方法检测异常值
        Args:
            series: 数据序列
            threshold: Z-score阈值
        Returns:
            异常值索引的布尔数组
        """
        if SCIPY_AVAILABLE:
            z_scores = np.abs(stats.zscore(series.dropna()))
        else:
            # 手动计算Z-score
            s = series.dropna()
            if len(s) > 0 and s.std() > 0:
                z_scores = np.abs((s - s.mean()) / s.std())
            else:
                z_scores = np.zeros(len(s))
        
        outlier_mask = pd.Series(False, index=series.index)
        if len(series.dropna()) > 0:
            outlier_mask.loc[series.dropna().index] = z_scores > threshold
        return outlier_mask
    
    def winsorize(self, series, lower_percentile=1, upper_percentile=99):
        """
        Winsorization技术：将异常值裁剪至合理范围
        Args:
            series: 数据序列
            lower_percentile: 下分位数
            upper_percentile: 上分位数
        Returns:
            处理后的序列
        """
        lower_bound = series.quantile(lower_percentile / 100)
        upper_bound = series.quantile(upper_percentile / 100)
        return series.clip(lower=lower_bound, upper=upper_bound)
    
    def handle_outliers(self, df, numeric_cols, method='winsorize', 
                       create_indicator=True, iqr_factor=1.5, zscore_threshold=3):
        """
        异常值检测与处理
        Args:
            df: 输入数据框
            numeric_cols: 数值型特征列表
            method: 处理方法 ('winsorize', 'quantile_clip', 'log_transform')
            create_indicator: 是否创建异常值指示变量
            iqr_factor: IQR因子
            zscore_threshold: Z-score阈值
        Returns:
            处理后的数据框
        """
        print(f"Handling outliers using {method} method...")
        df = df.copy()
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
                
            # 创建异常值指示变量
            if create_indicator:
                outlier_mask_iqr = self.detect_outliers_iqr(df[col], factor=iqr_factor)
                outlier_mask_zscore = self.detect_outliers_zscore(df[col], threshold=zscore_threshold)
                # 合并两种方法的检测结果
                outlier_mask = outlier_mask_iqr | outlier_mask_zscore
                df[f'{col}_is_outlier'] = outlier_mask.astype(int)
            
            # 处理异常值
            if method == 'winsorize':
                df[col] = self.winsorize(df[col])
            elif method == 'quantile_clip':
                # 分位数截断
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif method == 'log_transform':
                # 对数变换（仅对正值）
                min_val = df[col].min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    df[col] = np.log1p(df[col] + shift)
                else:
                    df[col] = np.log1p(df[col])
        
        return df
    
    def handle_missing_values(self, df, numeric_cols, cat_cols, 
                             method='knn', n_neighbors=5):
        """
        缺失值处理
        Args:
            df: 输入数据框
            numeric_cols: 数值型特征列表
            cat_cols: 分类型特征列表
            method: 插值方法 ('knn', 'mean', 'median', 'mode', 'keep_missing')
            n_neighbors: KNN插值的邻居数
        Returns:
            处理后的数据框
        """
        print(f"Handling missing values using {method} method...")
        df = df.copy()
        
        # 创建缺失值指示变量（保留缺失信息）
        for col in numeric_cols + cat_cols:
            if col not in df.columns:
                continue
            if df[col].isna().sum() > 0:
                df[f'{col}_is_missing'] = df[col].isna().astype(int)
        
        # 处理数值型特征的缺失值
        numeric_missing_cols = [col for col in numeric_cols if col in df.columns and df[col].isna().sum() > 0]
        
        if numeric_missing_cols:
            if method == 'knn':
                # KNN插值
                if not SKLEARN_AVAILABLE:
                    print(f"  Warning: KNN imputation not available, using mean instead...")
                    method = 'mean'
                else:
                    print(f"  Using KNN imputation for {len(numeric_missing_cols)} numeric features...")
                    imputer = KNNImputer(n_neighbors=n_neighbors)
                    df[numeric_missing_cols] = imputer.fit_transform(df[numeric_missing_cols])
                    self.imputers['knn_numeric'] = imputer
            elif method == 'mean':
                imputer = SimpleImputer(strategy='mean')
                df[numeric_missing_cols] = imputer.fit_transform(df[numeric_missing_cols])
                self.imputers['mean_numeric'] = imputer
            elif method == 'median':
                imputer = SimpleImputer(strategy='median')
                df[numeric_missing_cols] = imputer.fit_transform(df[numeric_missing_cols])
                self.imputers['median_numeric'] = imputer
            elif method == 'keep_missing':
                # 保留缺失值，仅创建指示变量
                pass
        
        # 处理分类型特征的缺失值
        cat_missing_cols = [col for col in cat_cols if col in df.columns and df[col].isna().sum() > 0]
        
        if cat_missing_cols:
            if method in ['mean', 'median', 'knn']:
                # 对于分类特征，使用众数填充
                imputer = SimpleImputer(strategy='most_frequent')
                df[cat_missing_cols] = imputer.fit_transform(df[cat_missing_cols])
                self.imputers['mode_categorical'] = imputer
            elif method == 'keep_missing':
                pass
        
        return df
    
    def normalize_features(self, df, numeric_cols, method='standardize', 
                          fit_on_train=True):
        """
        特征标准化与归一化
        Args:
            df: 输入数据框
            numeric_cols: 数值型特征列表
            method: 标准化方法 ('standardize', 'minmax', 'robust')
            fit_on_train: 是否拟合（训练集为True，测试集为False）
        Returns:
            处理后的数据框
        """
        print(f"Normalizing features using {method} method...")
        df = df.copy()
        
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        for col in numeric_cols:
            if fit_on_train:
                if method == 'standardize':
                    if SKLEARN_AVAILABLE:
                        scaler = StandardScaler()
                        df[col] = scaler.fit_transform(df[[col]]).flatten()
                        self.scalers[col] = scaler
                    else:
                        # 手动标准化
                        mean = df[col].mean()
                        std = df[col].std()
                        if std > 0:
                            df[col] = (df[col] - mean) / std
                        self.feature_stats[col] = {'mean': mean, 'std': std}
                elif method == 'minmax':
                    if SKLEARN_AVAILABLE:
                        scaler = MinMaxScaler()
                        df[col] = scaler.fit_transform(df[[col]]).flatten()
                        self.scalers[col] = scaler
                    else:
                        # 手动归一化
                        min_val = df[col].min()
                        max_val = df[col].max()
                        if max_val > min_val:
                            df[col] = (df[col] - min_val) / (max_val - min_val)
                        self.feature_stats[col] = {'min': min_val, 'max': max_val}
                elif method == 'robust':
                    # 使用中位数和IQR进行标准化（对异常值更稳健）
                    median = df[col].median()
                    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                    if iqr > 0:
                        df[col] = (df[col] - median) / iqr
                    self.feature_stats[col] = {'median': median, 'iqr': iqr}
            else:
                # 测试集：使用训练集拟合的scaler
                if col in self.scalers:
                    if method == 'standardize' or method == 'minmax':
                        df[col] = self.scalers[col].transform(df[[col]]).flatten()
                elif col in self.feature_stats:
                    # robust标准化
                    stats = self.feature_stats[col]
                    if stats['iqr'] > 0:
                        df[col] = (df[col] - stats['median']) / stats['iqr']
        
        return df
    
    def extract_time_features(self, df, time_col='S_2'):
        """
        提取时间特征（针对Amex数据集）
        Args:
            df: 输入数据框
            time_col: 时间列名
        Returns:
            添加了时间特征的数据框
        """
        if time_col not in df.columns:
            return df
        
        print("Extracting time features...")
        df = df.copy()
        
        # 确保时间列为datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # 提取基础时间特征
        df[f'{time_col}_year'] = df[time_col].dt.year
        df[f'{time_col}_month'] = df[time_col].dt.month
        df[f'{time_col}_day'] = df[time_col].dt.day
        df[f'{time_col}_dayofweek'] = df[time_col].dt.dayofweek
        df[f'{time_col}_quarter'] = df[time_col].dt.quarter
        
        return df
    
    def calculate_time_window_features(self, df, id_col='customer_ID', 
                                      time_col='S_2', value_cols=None, 
                                      windows=[3, 6]):
        """
        计算滑动窗口统计特征（用于时序数据平滑）
        Args:
            df: 输入数据框
            id_col: ID列
            time_col: 时间列
            value_cols: 需要计算窗口统计的数值列
            windows: 窗口大小列表（月数）
        Returns:
            添加了窗口特征的数据框
        """
        if value_cols is None or time_col not in df.columns:
            return df
        
        print(f"Calculating time window features for windows {windows}...")
        df = df.copy()
        
        # 确保按ID和时间排序
        df = df.sort_values([id_col, time_col])
        
        # 确保时间列为datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        for col in value_cols:
            if col not in df.columns:
                continue
            
            for window in windows:
                # 计算滚动窗口统计量
                window_days = window * 30  # 近似月数
                
                # 按客户分组计算滚动统计
                df[f'{col}_rolling_mean_{window}m'] = df.groupby(id_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f'{col}_rolling_std_{window}m'] = df.groupby(id_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
                )
        
        return df
    
    def preprocess(self, df, id_col='customer_ID', time_col=None, 
                   numeric_cols=None, cat_cols=None,
                   remove_dups=True, handle_missing=True, handle_outliers=True,
                   normalize=True, extract_time=True):
        """
        完整的数据预处理流程
        Args:
            df: 输入数据框
            id_col: ID列名
            time_col: 时间列名（如果有）
            numeric_cols: 数值型特征列表
            cat_cols: 分类型特征列表
            remove_dups: 是否去重
            handle_missing: 是否处理缺失值
            handle_outliers: 是否处理异常值
            normalize: 是否标准化
            extract_time: 是否提取时间特征
        Returns:
            预处理后的数据框
        """
        print("=" * 60)
        print("Starting Data Preprocessing Pipeline")
        print("=" * 60)
        
        df = df.copy()
        
        # 自动识别特征类型（如果未提供）
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if id_col in numeric_cols:
                numeric_cols.remove(id_col)
            if time_col and time_col in numeric_cols:
                numeric_cols.remove(time_col)
        
        if cat_cols is None:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if time_col and time_col in cat_cols:
                cat_cols.remove(time_col)
        
        # 1. 去重
        if remove_dups:
            df = self.remove_duplicates(df, id_col=id_col, time_col=time_col)
        
        # 2. 提取时间特征
        if extract_time and time_col:
            df = self.extract_time_features(df, time_col=time_col)
        
        # 3. 处理异常值
        if handle_outliers:
            df = self.handle_outliers(df, numeric_cols, method='winsorize')
        
        # 4. 处理缺失值
        if handle_missing:
            df = self.handle_missing_values(df, numeric_cols, cat_cols, method='knn')
        
        # 5. 标准化
        if normalize:
            df = self.normalize_features(df, numeric_cols, method='standardize')
        
        print("=" * 60)
        print("Data Preprocessing Complete")
        print("=" * 60)
        
        return df
