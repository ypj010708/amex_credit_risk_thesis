"""
特征工程主流程
整合数据预处理、统计视图构建、时序视图构建和特征选择
实现论文第3章描述的完整特征工程流程
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')

from .preprocessor import DataPreprocessor
from .features_statistical import (
    build_time_window_features,
    build_trend_features,
    build_agg_features,
    build_lendingclub_features
)
from .features_sequential import SequentialFeatureExtractor
from .selection import cascade_feature_selection
from .loader import load_parquet_data, reduce_mem_usage


class FeatureEngineeringPipeline:
    """特征工程主流程类"""
    
    def __init__(self, dataset_type='amex', config_path=None):
        """
        Args:
            dataset_type: 'amex' 或 'lendingclub'
            config_path: 配置文件路径
        """
        self.dataset_type = dataset_type
        self.preprocessor = DataPreprocessor(dataset_type=dataset_type)
        self.sequential_extractor = None
        self.selected_features = None
        
        # 加载配置
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
    
    def load_data(self, train_path, test_path=None, sample_size=None):
        """
        加载数据
        Args:
            train_path: 训练数据路径
            test_path: 测试数据路径（可选）
            sample_size: 采样大小（用于快速测试）
        Returns:
            train_df, test_df
        """
        print("=" * 60)
        print("Loading Data")
        print("=" * 60)
        
        # 加载训练集
        if train_path.endswith('.parquet'):
            # 优先使用fastparquet避免pyarrow DLL问题
            train_df = load_parquet_data(train_path, engine='fastparquet')
        else:
            train_df = pd.read_csv(train_path, low_memory=False)
        
        if sample_size and len(train_df) > sample_size:
            train_df = self._sample_dataset(
                train_df,
                sample_size=sample_size,
                id_col='customer_ID' if 'customer_ID' in train_df.columns else None,
                target_col='target' if 'target' in train_df.columns else None
            )

        # 如果训练集中没有 target 列，尝试自动加载标签文件并合并（仅针对 amex 数据集）
        if 'target' not in train_df.columns and self.dataset_type == 'amex':
            print("提示: 训练数据中未找到 'target' 列，尝试从独立标签文件中加载并合并...")
            train_path_obj = Path(train_path)
            candidate_label_files = [
                train_path_obj.with_name('train_labels.csv'),
                train_path_obj.with_name('train_labels.parquet'),
                Path('data/raw/train_labels.csv'),
                Path('data/raw/train_labels.parquet'),
            ]
            label_df = None
            for lp in candidate_label_files:
                if lp.exists():
                    print(f"  尝试从 {lp} 加载标签...")
                    try:
                        if lp.suffix == '.csv':
                            tmp = pd.read_csv(lp, low_memory=False)
                        else:
                            tmp = load_parquet_data(str(lp), engine='fastparquet')
                        # 需要包含 customer_ID 和 target 两列
                        if 'customer_ID' in tmp.columns and 'target' in tmp.columns:
                            label_df = tmp[['customer_ID', 'target']].drop_duplicates('customer_ID')
                            break
                        else:
                            print(f"  文件 {lp} 中不包含 'customer_ID' 和 'target' 列，跳过。")
                    except Exception as e:
                        print(f"  从 {lp} 加载标签失败: {e}")

            if label_df is not None and 'customer_ID' in train_df.columns:
                before_cols = set(train_df.columns)
                train_df = train_df.merge(label_df, on='customer_ID', how='left')
                after_cols = set(train_df.columns)
                print(f"  标签合并完成，新列: {sorted(list(after_cols - before_cols))}")
                if train_df['target'].isna().any():
                    n_missing = int(train_df['target'].isna().sum())
                    print(f"  警告: 有 {n_missing} 个 customer_ID 在标签文件中找不到，对应 target 为 NaN。")
            else:
                print("  未找到可用的标签文件，后续将无法计算IV或进行有监督特征选择。")
        
        # 加载测试集（如果有）
        test_df = None
        if test_path:
            if test_path.endswith('.parquet'):
                # 优先使用fastparquet避免pyarrow DLL问题
                test_df = load_parquet_data(test_path, engine='fastparquet')
            else:
                test_df = pd.read_csv(test_path, low_memory=False)
            
            if sample_size and len(test_df) > sample_size:
                test_df = self._sample_dataset(
                    test_df,
                    sample_size=sample_size,
                    id_col='customer_ID' if 'customer_ID' in test_df.columns else None,
                    target_col='target' if 'target' in test_df.columns else None
                )

        print(f"Train data shape: {train_df.shape}")
        if test_df is not None:
            print(f"Test data shape: {test_df.shape}")
        
        return train_df, test_df

    def _sample_dataset(self, df, sample_size, id_col=None, target_col=None):
        """
        对数据集进行采样，优先按客户、按标签分层采样，保证类别平衡
        """
        if sample_size is None or len(df) <= sample_size:
            return df

        if id_col and id_col in df.columns:
            customer_targets = df.groupby(id_col)[target_col].first() if (target_col and target_col in df.columns) else None
            unique_ids = df[id_col].unique()
            avg_records = max(1, len(df) // max(1, len(unique_ids)))
            n_customers = max(1, min(len(unique_ids), sample_size // avg_records))

            if customer_targets is not None and customer_targets.nunique() > 1:
                sampled_ids = []
                remaining = n_customers
                groups = customer_targets.groupby(customer_targets)
                total_customers = len(customer_targets)
                for value, group in groups:
                    n_needed = max(1, int(round(n_customers * len(group) / total_customers)))
                    n_needed = min(n_needed, len(group))
                    remaining -= n_needed
                    sampled_ids.append(group.sample(n=n_needed, random_state=42))
                if remaining > 0:
                    leftover = customer_targets.drop(pd.concat(sampled_ids).index)
                    if not leftover.empty:
                        sampled_ids.append(leftover.sample(n=min(remaining, len(leftover)), random_state=42))
                sampled_ids = pd.Index(pd.concat(sampled_ids).index.unique())
            else:
                sampled_ids = pd.Index(pd.Series(unique_ids).sample(n=n_customers, random_state=42).values)

            df_sampled = df[df[id_col].isin(sampled_ids)].copy()
            return df_sampled

        # 非客户级别数据，直接按标签分层采样
        if target_col and target_col in df.columns and df[target_col].nunique() > 1:
            frac = sample_size / len(df)
            df_sampled = df.groupby(target_col, group_keys=False).apply(
                lambda x: x.sample(n=max(1, int(round(len(x) * frac))), random_state=42)
            ).reset_index(drop=True)
            return df_sampled
        
        return df.sample(n=sample_size, random_state=42)
    
    def preprocess_data(self, train_df, test_df=None, 
                       id_col=None, time_col=None):
        """
        数据预处理与清洗（论文3.2.1节）
        Args:
            train_df: 训练数据
            test_df: 测试数据（可选）
            id_col: ID列名
            time_col: 时间列名
        Returns:
            预处理后的数据框
        """
        print("=" * 60)
        print("Data Preprocessing (Section 3.2.1)")
        print("=" * 60)
        
        # 自动识别ID列和时间列
        if id_col is None:
            id_col = 'customer_ID' if 'customer_ID' in train_df.columns else 'id'
        if time_col is None:
            time_col = 'S_2' if 'S_2' in train_df.columns else None
        
        # 识别特征类型
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if id_col in numeric_cols:
            numeric_cols.remove(id_col)
        if time_col and time_col in numeric_cols:
            numeric_cols.remove(time_col)
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
        if time_col and time_col in cat_cols:
            cat_cols.remove(time_col)
        
        # 预处理训练集
        train_df_processed = self.preprocessor.preprocess(
            train_df,
            id_col=id_col,
            time_col=time_col,
            numeric_cols=numeric_cols,
            cat_cols=cat_cols
        )
        
        # 预处理测试集（使用训练集的scaler）
        if test_df is not None:
            test_df_processed = self.preprocessor.preprocess(
                test_df,
                id_col=id_col,
                time_col=time_col,
                numeric_cols=numeric_cols,
                cat_cols=cat_cols
            )
        else:
            test_df_processed = None
        
        return train_df_processed, test_df_processed, id_col, time_col
    
    def build_statistical_features(self, train_df, test_df=None,
                                  id_col='customer_ID', time_col='S_2'):
        """
        构建统计视图特征（论文3.2.2节）
        包括：时窗聚合特征、趋势斜率特征、基础统计聚合
        Args:
            train_df: 训练数据
            test_df: 测试数据（可选）
            id_col: ID列
            time_col: 时间列
        Returns:
            统计特征数据框
        """
        print("=" * 60)
        print("Building Statistical View Features (Section 3.2.2)")
        print("=" * 60)
        
        # 识别数值特征
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols 
                       if col not in [id_col, time_col, 'target']]
        
        cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_cols = [col for col in cat_cols if col not in [id_col, time_col]]
        
        # 1. 时窗聚合特征
        print("\n1. Building time window aggregation features...")
        train_window = build_time_window_features(
            train_df, id_col=id_col, time_col=time_col,
            value_cols=numeric_cols[:20],  # 限制特征数量以加快速度
            windows=[3, 6]
        )
        
        if test_df is not None:
            test_window = build_time_window_features(
                test_df, id_col=id_col, time_col=time_col,
                value_cols=numeric_cols[:20],
                windows=[3, 6]
            )
        else:
            test_window = None
        
        # 2. 趋势斜率特征
        print("\n2. Building trend slope features...")
        # 选择关键特征计算趋势
        key_features = numeric_cols[:10]  # 选择前10个关键特征
        train_trend = build_trend_features(
            train_df, target_features=key_features,
            id_col=id_col, time_col=time_col,
            time_windows=[3, 6]
        )
        
        if test_df is not None:
            test_trend = build_trend_features(
                test_df, target_features=key_features,
                id_col=id_col, time_col=time_col,
                time_windows=[3, 6]
            )
        else:
            test_trend = None
        
        # 3. 基础统计聚合
        print("\n3. Building standard aggregation features...")
        train_agg = build_agg_features(
            train_df, num_features=numeric_cols, 
            cat_features=cat_cols, id_col=id_col
        )
        
        if test_df is not None:
            test_agg = build_agg_features(
                test_df, num_features=numeric_cols,
                cat_features=cat_cols, id_col=id_col
            )
        else:
            test_agg = None
        
        # 合并所有统计特征
        print("\n4. Merging statistical features...")
        train_stat = train_agg.merge(train_window, on=id_col, how='left')
        train_stat = train_stat.merge(train_trend, on=id_col, how='left')
        train_stat = train_stat.fillna(0)
        
        if test_df is not None:
            test_stat = test_agg.merge(test_window, on=id_col, how='left')
            test_stat = test_stat.merge(test_trend, on=id_col, how='left')
            test_stat = test_stat.fillna(0)
        else:
            test_stat = None
        
        print(f"Statistical features shape: {train_stat.shape}")
        
        return train_stat, test_stat
    
    def build_sequential_features(self, train_df, test_df=None,
                                 id_col='customer_ID', time_col='S_2',
                                 model_type='transformer', epochs=5):
        """
        构建时序视图特征（论文3.2.3节）
        使用GRU或Transformer提取时序嵌入
        Args:
            train_df: 训练数据
            test_df: 测试数据（可选）
            id_col: ID列
            time_col: 时间列
            model_type: 'gru' 或 'transformer'
            epochs: 训练轮数
        Returns:
            时序嵌入特征数据框
        """
        print("=" * 60)
        print("Building Sequential View Features (Section 3.2.3)")
        print("=" * 60)
        
        # 初始化时序提取器
        self.sequential_extractor = SequentialFeatureExtractor(
            model_type=model_type,
            hidden_dim=128,
            device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
        )
        
        # 识别特征列
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols 
                       if col not in [id_col, time_col, 'target']]
        
        # 提取训练集时序嵌入
        print("\nExtracting sequential embeddings for training set...")
        if 'target' in train_df.columns:
            train_seq = self.sequential_extractor.fit_transform(
                train_df, id_col=id_col, time_col=time_col,
                feature_cols=feature_cols[:50],  # 限制特征数量
                target_col='target',
                epochs=epochs
            )
        else:
            self.sequential_extractor.fit(
                train_df, id_col=id_col, time_col=time_col,
                feature_cols=feature_cols[:50]
            )
            train_seq = self.sequential_extractor.transform(
                train_df, id_col=id_col, time_col=time_col,
                feature_cols=feature_cols[:50]
            )
        
        # 提取测试集时序嵌入
        if test_df is not None:
            print("\nExtracting sequential embeddings for test set...")
            test_seq = self.sequential_extractor.transform(
                test_df, id_col=id_col, time_col=time_col,
                feature_cols=feature_cols[:50]
            )
        else:
            test_seq = None
        
        print(f"Sequential features shape: {train_seq.shape}")
        
        return train_seq, test_seq
    
    def select_features(self, X_train, y_train, X_val=None,
                       min_iv=0.02, max_psi=0.25, use_null_importance=True):
        """
        级联式特征选择（论文3.3节）
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征（可选）
            min_iv: IV最小阈值
            max_psi: PSI最大阈值
            use_null_importance: 是否使用Null Importance
        Returns:
            选中的特征列表
        """
        print("=" * 60)
        print("Cascade Feature Selection (Section 3.3)")
        print("=" * 60)
        
        selected = cascade_feature_selection(
            X_train, y_train, X_val=X_val,
            min_iv=min_iv, max_psi=max_psi,
            null_importance=use_null_importance
        )
        
        self.selected_features = selected
        return selected
    
    def run_full_pipeline(self, train_path, test_path=None,
                         target_col='target', sample_size=None,
                         build_sequential=True, select_features=True):
        """
        运行完整的特征工程流程
        Args:
            train_path: 训练数据路径
            test_path: 测试数据路径（可选）
            target_col: 目标列名
            sample_size: 采样大小
            build_sequential: 是否构建时序特征
            select_features: 是否进行特征选择
        Returns:
            最终特征数据框
        """
        print("=" * 80)
        print("FEATURE ENGINEERING PIPELINE")
        print("=" * 80)
        
        # 1. 加载数据
        train_df, test_df = self.load_data(train_path, test_path, sample_size)
        
        # 2. 数据预处理
        train_df_proc, test_df_proc, id_col, time_col = self.preprocess_data(
            train_df, test_df
        )
        
        # 3. 构建统计视图特征
        train_stat, test_stat = self.build_statistical_features(
            train_df_proc, test_df_proc, id_col=id_col, time_col=time_col
        )
        
        # 4. 构建时序视图特征（可选）
        if build_sequential and time_col:
            train_seq, test_seq = self.build_sequential_features(
                train_df_proc, test_df_proc, id_col=id_col, time_col=time_col
            )
            
            # 合并时序特征
            train_stat = train_stat.merge(train_seq, on=id_col, how='left')
            if test_stat is not None:
                test_stat = test_stat.merge(test_seq, on=id_col, how='left')
        else:
            print("Skipping sequential features (no time column or disabled)")
        
        # 5. 提取目标变量
        if target_col in train_df.columns:
            y_train = train_df.groupby(id_col)[target_col].first().reindex(
                train_stat[id_col]
            ).values
        else:
            y_train = None
        
        # 6. 特征选择（如果有标签）
        if select_features and y_train is not None:
            # 准备特征矩阵（排除ID列）
            feature_cols = [col for col in train_stat.columns if col != id_col]
            X_train = train_stat[feature_cols]
            
            # 特征选择
            selected = self.select_features(X_train, y_train)
            
            # 筛选特征
            final_cols = [id_col] + selected
            train_stat = train_stat[final_cols]
            if test_stat is not None:
                test_stat = test_stat[final_cols]
        else:
            print("Skipping feature selection (no target or disabled)")
        
        # 内存优化
        train_stat = reduce_mem_usage(train_stat)
        if test_stat is not None:
            test_stat = reduce_mem_usage(test_stat)
        
        print("=" * 80)
        print("FEATURE ENGINEERING COMPLETE")
        print(f"Final feature shape: {train_stat.shape}")
        print("=" * 80)
        
        return train_stat, test_stat, y_train


def main():
    """主函数示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature Engineering Pipeline')
    parser.add_argument('--dataset', type=str, default='amex',
                       choices=['amex', 'lendingclub'],
                       help='Dataset type')
    parser.add_argument('--train_path', type=str, required=True,
                       help='Training data path')
    parser.add_argument('--test_path', type=str, default=None,
                       help='Test data path (optional)')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='Output directory')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size for quick testing')
    parser.add_argument('--no_sequential', action='store_true',
                       help='Skip sequential feature extraction')
    parser.add_argument('--no_selection', action='store_true',
                       help='Skip feature selection')
    
    args = parser.parse_args()
    
    # 创建特征工程管道
    pipeline = FeatureEngineeringPipeline(dataset_type=args.dataset)
    
    # 运行完整流程
    train_features, test_features, y_train = pipeline.run_full_pipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        sample_size=args.sample_size,
        build_sequential=not args.no_sequential,
        select_features=not args.no_selection
    )
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_features.to_parquet(output_dir / 'train_features.parquet', index=False)
    if test_features is not None:
        test_features.to_parquet(output_dir / 'test_features.parquet', index=False)
    
    print(f"\nFeatures saved to {output_dir}")


if __name__ == '__main__':
    main()

