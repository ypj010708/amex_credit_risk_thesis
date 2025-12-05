"""
时序视图构建模块
实现论文3.2.3节描述的时序嵌入与异构特征融合
使用GRU和Transformer提取时序隐向量
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """时序数据数据集类"""
    
    def __init__(self, sequences, static_features=None, targets=None):
        """
        Args:
            sequences: 时序数据 (n_samples, seq_len, n_features)
            static_features: 静态特征 (n_samples, n_static_features)
            targets: 目标变量 (n_samples,)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.static_features = torch.FloatTensor(static_features) if static_features is not None else None
        self.targets = torch.FloatTensor(targets) if targets is not None else None
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        item = {'sequence': self.sequences[idx]}
        if self.static_features is not None:
            item['static'] = self.static_features[idx]
        if self.targets is not None:
            item['target'] = self.targets[idx]
        return item


class GRUEncoder(nn.Module):
    """GRU时序编码器"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.1):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: GRU层数
            dropout: Dropout比率
        """
        super(GRUEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, hidden_dim) - 最后一个时间步的隐状态
        """
        # x shape: (batch_size, seq_len, input_dim)
        gru_out, hidden = self.gru(x)
        # 使用最后一个时间步的输出
        output = gru_out[:, -1, :]  # (batch_size, hidden_dim)
        return output


class TransformerEncoder(nn.Module):
    """Transformer时序编码器（基于自注意力机制）"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=2, 
                 dim_feedforward=512, dropout=0.1, max_seq_len=13):
        """
        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            max_seq_len: 最大序列长度
        """
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(max_seq_len, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
        Returns:
            output: (batch_size, d_model) - 全局池化后的表示
        """
        batch_size, seq_len, _ = x.shape
        
        # 投影到d_model维度
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # 添加位置编码
        x = x + self.pos_encoder[:seq_len, :].unsqueeze(0)
        
        # Transformer编码
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # 全局平均池化
        output = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # 输出投影
        output = self.output_projection(output)
        
        return output


class SequentialFeatureExtractor:
    """时序特征提取器"""
    
    def __init__(self, model_type='transformer', hidden_dim=128, device='cpu'):
        """
        Args:
            model_type: 'gru' 或 'transformer'
            hidden_dim: 隐藏层/模型维度
            device: 计算设备
        """
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.model = None
        self.is_fitted = False
        
    def _prepare_sequences(self, df, id_col, time_col, feature_cols):
        """
        准备时序数据
        Args:
            df: 输入数据框
            id_col: ID列
            time_col: 时间列
            feature_cols: 特征列列表
        Returns:
            sequences: (n_samples, seq_len, n_features) 数组
            ids: ID列表
        """
        print("Preparing time series sequences...")
        
        df = df.copy()
        
        # 确保时间列为datetime类型
        if time_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            df = df.sort_values([id_col, time_col])
        else:
            df = df.sort_values([id_col])
        
        sequences = []
        ids = []
        
        for uid, group in tqdm(df.groupby(id_col), desc="Preparing Sequences"):
            # 提取特征值
            feature_values = group[feature_cols].values.astype(np.float32)
            
            # 填充缺失值
            feature_values = pd.DataFrame(feature_values).fillna(0).values
            
            sequences.append(feature_values)
            ids.append(uid)
        
        # 找到最大序列长度
        max_len = max(len(seq) for seq in sequences)
        
        # 填充序列到相同长度
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_len:
                # 使用零填充
                padding = np.zeros((max_len - len(seq), len(feature_cols)))
                padded_seq = np.vstack([seq, padding])
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        
        sequences_array = np.array(padded_sequences)
        print(f"Sequences shape: {sequences_array.shape}")
        
        return sequences_array, ids
    
    def fit(self, df, id_col='customer_ID', time_col='S_2', 
            feature_cols=None, static_cols=None, target_col='target',
            epochs=10, batch_size=64, lr=0.001):
        """
        训练时序编码器（可选：如果有标签可以微调）
        Args:
            df: 输入数据框
            id_col: ID列
            time_col: 时间列
            feature_cols: 时序特征列列表
            static_cols: 静态特征列列表
            target_col: 目标列（可选，用于监督学习）
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
        """
        print(f"Fitting {self.model_type} encoder...")
        
        # 自动识别特征列
        if feature_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols 
                           if col not in [id_col, time_col, target_col]]
        
        # 准备时序数据
        sequences, ids = self._prepare_sequences(df, id_col, time_col, feature_cols)
        
        input_dim = len(feature_cols)
        
        # 初始化模型
        if self.model_type == 'gru':
            self.model = GRUEncoder(input_dim, hidden_dim=self.hidden_dim).to(self.device)
        elif self.model_type == 'transformer':
            self.model = TransformerEncoder(input_dim, d_model=self.hidden_dim).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # 如果有标签，可以进行监督学习微调
        if target_col in df.columns:
            print("Performing supervised fine-tuning...")
            targets = df.groupby(id_col)[target_col].first().reindex(ids).values
            
            # 准备静态特征（如果有）
            static_features = None
            if static_cols:
                static_df = df.groupby(id_col)[static_cols].first().reindex(ids)
                static_features = static_df.fillna(0).values.astype(np.float32)
            
            # 创建数据集和数据加载器
            dataset = TimeSeriesDataset(sequences, static_features, targets)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # 定义优化器和损失函数
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            
            # 训练循环
            self.model.train()
            for epoch in range(epochs):
                total_loss = 0
                for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                    optimizer.zero_grad()
                    
                    seq_emb = self.model(batch['sequence'].to(self.device))
                    
                    # 如果有静态特征，进行融合
                    if batch['static'] is not None:
                        static_emb = batch['static'].to(self.device)
                        # 简单拼接融合
                        combined = torch.cat([seq_emb, static_emb], dim=1)
                        # 添加一个线性层进行预测
                        if not hasattr(self, 'predictor'):
                            self.predictor = nn.Linear(
                                combined.shape[1], 1
                            ).to(self.device)
                        logits = self.predictor(combined).squeeze()
                    else:
                        if not hasattr(self, 'predictor'):
                            self.predictor = nn.Linear(self.hidden_dim, 1).to(self.device)
                        logits = self.predictor(seq_emb).squeeze()
                    
                    loss = criterion(logits, batch['target'].to(self.device))
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
        else:
            # 无监督学习：直接使用模型提取特征
            print("Using unsupervised feature extraction...")
        
        self.is_fitted = True
        print("Model fitting complete.")
    
    def transform(self, df, id_col='customer_ID', time_col='S_2', 
                  feature_cols=None, static_cols=None):
        """
        提取时序嵌入特征
        Args:
            df: 输入数据框
            id_col: ID列
            time_col: 时间列
            feature_cols: 时序特征列列表
            static_cols: 静态特征列列表（用于融合）
        Returns:
            嵌入特征数据框（每个ID一行）
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform. Call fit() first.")
        
        print(f"Extracting {self.model_type} embeddings...")
        
        # 自动识别特征列
        if feature_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols 
                           if col not in [id_col, time_col]]
        
        # 准备时序数据
        sequences, ids = self._prepare_sequences(df, id_col, time_col, feature_cols)
        
        # 创建数据集
        static_features = None
        if static_cols:
            static_df = df.groupby(id_col)[static_cols].first().reindex(ids)
            static_features = static_df.fillna(0).values.astype(np.float32)
        
        dataset = TimeSeriesDataset(sequences, static_features)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        
        # 提取嵌入
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting Embeddings"):
                seq_emb = self.model(batch['sequence'].to(self.device))
                
                # 如果有静态特征，进行融合
                if batch['static'] is not None:
                    static_emb = batch['static'].to(self.device)
                    # 加权融合（可以调整权重）
                    combined = torch.cat([seq_emb, static_emb], dim=1)
                    embeddings.append(combined.cpu().numpy())
                else:
                    embeddings.append(seq_emb.cpu().numpy())
        
        # 合并所有嵌入
        all_embeddings = np.vstack(embeddings)
        
        # 创建特征数据框
        feature_names = [f'{self.model_type}_emb_{i}' for i in range(all_embeddings.shape[1])]
        df_emb = pd.DataFrame(all_embeddings, columns=feature_names)
        df_emb[id_col] = ids
        
        print(f"Embedding extraction complete. Shape: {df_emb.shape}")
        return df_emb
    
    def fit_transform(self, df, id_col='customer_ID', time_col='S_2',
                     feature_cols=None, static_cols=None, target_col='target',
                     epochs=10, batch_size=64, lr=0.001):
        """拟合并转换"""
        self.fit(df, id_col, time_col, feature_cols, static_cols, 
                target_col, epochs, batch_size, lr)
        return self.transform(df, id_col, time_col, feature_cols, static_cols)
