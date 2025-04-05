__author__ = 'minjinwu'

import torch
import torch.nn as nn

# 定义 TransformerEncoderLayer 类
class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ff_dim):
        super(TransformerEncoderLayer, self).__init__()
        
        self.multihead_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, model_dim)
        )
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.multihead_attention(x, x, x, attn_mask=mask)
        x = self.layer_norm1(attn_output + x)  # Residual connection and layer normalization
        
        # Feedforward network with residual connection
        ffn_output = self.feedforward(x)
        x = self.layer_norm2(ffn_output + x)  # Residual connection and layer normalization
        
        return x

if __name__ == '__main__':
    # 设置超参数
    model_dim = 512  # 模型的维度
    num_heads = 8  # 多头注意力的头数
    ff_dim = 2048  # 前馈网络的维度
    batch_size = 32  # 批次大小
    seq_length = 50  # 序列长度

    # 创建输入数据
    x = torch.rand(seq_length, batch_size, model_dim)  # 输入的形状为 (seq_length, batch_size, model_dim)

    # 初始化 TransformerEncoderLayer
    encoder_layer = TransformerEncoderLayer(model_dim, num_heads, ff_dim)

    # 进行前向传播
    output = encoder_layer(x)

    # 打印输出的形状
    print(f"输出形状: {output.shape}")