import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.attention import MultiHeadAttention

# ==========================================
# 核心创新点实现：ATAE-GAN 模型架构
# ==========================================

class AttentionTabularEncoder(nn.Module):
    """
    带有注意力机制的表格编码器 (ATAE - Encoder 部分)。
    将高维表格数据映射到低维潜在空间 (Latent Space)，并在过程中利用 Attention 捕捉特征交互。
    """
    def __init__(self, input_dim, latent_dim=128, attn_embed_dim=64, num_heads=4, seq_len_for_attn=8):
        super().__init__()
        
        self.attn_embed_dim = attn_embed_dim
        self.seq_len = seq_len_for_attn
        # 计算需要的中间维度：为了 reshape 成 (seq_len, attn_embed_dim)
        self.inter_dim = self.seq_len * self.attn_embed_dim
        
        # 1. 初始特征提取与降维映射
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, self.inter_dim),
            nn.BatchNorm1d(self.inter_dim),
            nn.LeakyReLU(0.2)
        )
        
        # 2. 自注意力层：捕捉特征组之间的交互
        self.attention = MultiHeadAttention(embed_dim=attn_embed_dim, num_heads=num_heads)
        
        # 3. 映射到最终隐空间
        self.fc_out = nn.Linear(self.inter_dim, latent_dim)
        # 使用 Tanh 将隐向量约束在 [-1, 1] 之间 (常见于 GAN)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: (batch, input_dim)
        
        # 映射到中间维度
        x = self.fc_in(x) # (batch, inter_dim)
        
        # 重塑为 3D 张量以供注意力层使用: (Batch, Seq_Len, Embed_Dim)
        # 这里创造了一个人工的"序列"概念，让模型去发现不同特征块之间的关系
        x_reshaped = x.view(-1, self.seq_len, self.attn_embed_dim)
        
        # Self-Attention 交互
        x_attn = self.attention(x_reshaped) # (Batch, Seq, Dim)
        
        # 展平
        x_flat = x_attn.view(x.size(0), -1) # (Batch, inter_dim)
        
        # 映射到最终隐空间
        latent = self.tanh(self.fc_out(x_flat)) # (Batch, latent_dim)
        return latent

class TabularDecoder(nn.Module):
    """简单的解码器，将隐向量还原为表格数据"""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            # 假设输入数据已归一化到 [-1, 1] 或 [0, 1]
            # 如果是 Z-score 标准化，这里可能不需要激活函数或用 Tanh
            nn.Tanh() 
        )
    def forward(self, z):
        return self.model(z)

class AttentionTabularAutoEncoder(nn.Module):
    """
    将 Attention 编码器与解码器整合成自编码器，用于 Stage-1 预训练。
    """
    def __init__(self, input_dim, latent_dim=128, attn_embed_dim=64, num_heads=4, seq_len_for_attn=8):
        super().__init__()
        self.encoder = AttentionTabularEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            attn_embed_dim=attn_embed_dim,
            num_heads=num_heads,
            seq_len_for_attn=seq_len_for_attn,
        )
        self.decoder = TabularDecoder(latent_dim=latent_dim, output_dim=input_dim)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

class Critic(nn.Module):
    """
    WGAN-GP 的判别器 (称为 Critic)。
    评估输入样本是真实的还是生成的。
    注意：最后一层没有 Sigmoid 激活函数，输出的是无界的 Wasserstein 分数。
    """
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.model(x)

class ATAEGenerator(nn.Module):
    """
    基于自编码器解码器的生成器。先将噪声映射到隐空间，再解码到特征空间。
    """
    def __init__(self, decoder: TabularDecoder, noise_dim=128, latent_dim=128):
        super().__init__()
        self.decoder = decoder
        self.noise_to_latent = nn.Sequential(
            nn.Linear(noise_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )

    def forward(self, noise):
        latent = self.noise_to_latent(noise)
        return self.decoder(latent)

class BoundaryClassifier(nn.Module):
    """
    分类器反馈模块：提供决策边界信息，引导生成器关注难样本。
    """
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
