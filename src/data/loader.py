import pandas as pd
import numpy as np
import gc

def reduce_mem_usage(df, verbose=True):
    """
    遍历 DataFrame 的所有列并修改数据类型以减少内存使用。
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                # 尝试降级为 int8, int16, int32
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                # 浮点数降级为 float32
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage decreased to {end_mem:.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def load_parquet_data(path, columns=None, optimize_mem=True, engine='fastparquet'):
    """加载 parquet 数据并进行内存优化"""
    print(f"Loading data from {path}...")
    try:
        # 尝试不同的引擎
        engines_to_try = []
        if engine == 'auto':
            # 按优先级尝试
            try:
                import pyarrow
                engines_to_try.append('pyarrow')
            except:
                pass
            try:
                import fastparquet
                engines_to_try.append('fastparquet')
            except:
                pass
            if not engines_to_try:
                engines_to_try = [None]  # 使用pandas默认
        else:
            engines_to_try = [engine]
        
        df = None
        last_error = None
        for eng in engines_to_try:
            try:
                if eng:
                    df = pd.read_parquet(path, columns=columns, engine=eng)
                else:
                    df = pd.read_parquet(path, columns=columns)
                print(f"Successfully loaded using engine: {eng or 'default'}")
                break
            except Exception as e:
                last_error = e
                print(f"Failed to load with engine {eng}: {e}")
                continue
        
        if df is None:
            raise last_error or Exception("Failed to load parquet file with any engine")
        
        if optimize_mem:
            df = reduce_mem_usage(df)
        return df
    except FileNotFoundError:
        print(f"Error: The file {path} was not found.")
        return None
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        return None
    finally:
        gc.collect()
