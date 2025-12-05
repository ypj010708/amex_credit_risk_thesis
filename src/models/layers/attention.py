import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 核心创新点实现：PyTorch Self-Attention 模块
# ==========================================

class MultiHeadAttention(nn.Module):
    """
    标准多头自注意力机制 (Self-Attention) 的 PyTorch 实现封装。
    用于捕捉表格数据中不同特征组（在此处被视为一个"序列"）之间的交互。
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Args:
            embed_dim (int): 输入的嵌入维度。
            num_heads (int): 注意力头的数量。embed_dim 必须能被 num_heads 整除。
            dropout (float): Dropout 比率.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 使用 PyTorch 内置的高效实现，batch_first=True 表示输入格式为 (batch, seq, feature)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, 
                                         dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, embed_dim)
        Returns:
            torch.Tensor: 注意力处理后的张量，形状同输入，加入了残差连接和层归一化。
        """
        # 自注意力：Query, Key, Value 都是 x
        # attn_output shape: (batch, seq_len, embed_dim)
        attn_output, _ = self.mha(query=x, key=x, value=x)
        
        # 残差连接 (Residual Connection) 和 层归一化 (Layer Normalization)
        # 有助于梯度传播和训练稳定
        output = self.layer_norm(x + attn_output)
        
        return output
