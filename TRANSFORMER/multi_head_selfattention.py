__author__ = 'minjinwu'


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


    ## 多头注意力机制
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        # self.d_model = d_model
        # self.num_heads = num_heads
        # self.depth = d_model // num_heads  # 每个头的维度

        # Q, K, V 的线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出线性层
        self.W_o = nn.Linear(d_model, d_model)

        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)


    def split_heads(self, x, batch_size):
        # 将最后一维 d_model 拆成 num_heads * depth，并调整维度为 [batch, heads, seq_len, depth]
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, depth]


    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: [batch, heads, seq_len, depth]
        #attention_output = softmax(Q*K/sqrt(d_k))*V
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=Q.device))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output


    def forward(self, Q, K=None, V=None, mask=None):
        """
        通用前向传播接口，支持：
        - 自注意力：仅传入 Q (即x)
        - 编码器-解码器交叉注意力：传入 Q, K, V

        Q, K, V: [batch_size, seq_len, d_model]
        mask: [batch_size, 1, 1, seq_len] 或其他可广播形状
        """
        # 如果没有传入 K, V，说明是自注意力，Q=K=V
        if K is None:
            K = Q
        if V is None:
            V = Q

        batch_size = Q.size(0)
        residual = Q  # 残差连接，这里面是保存了线性变换之前的Q

        # 线性变换
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # 拆分多头
        "Q, K, V: [batch_size, heads, seq_len, depth]"
        "后续计算需要将多个头的维度进行合并"

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 计算注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头输出
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # 调整维度为[batch_size, seq_len, num_heads, depth]，并转换为连续张量
        # print("attn_output_multiheads:", attn_output.shape)
        # print("attn_output_multiheads:", attn_output[1, :, :, :])
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        # print("attn_output_mergedheads:", attn_output.shape)
        # print("attn_output_mergedheads:", attn_output[1, :, :])

        # 输出线性层
        output = self.W_o(attn_output)

        # 残差连接 + 层归一化（注意 residual 是原始 Q）
        output = self.layer_norm(output + residual)

        return output
