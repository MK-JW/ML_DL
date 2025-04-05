__author__ = 'minjinwu'


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


## 多头注意力机制
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        """
        多头注意力机制。
        
        参数：
        - d_model: 词向量的维度（例如 512) 类似于词嵌入的维度
        - num_heads: 头的数量（例如 8)
        """

        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除" # （这里面是要做平均分配每个头的维度，所以会要求被整除）
        
        """
        每个头的维度一致会带来更良好的计算,但是如果每个头分配的维度不是一致的呢？？？

        可以考虑每个人大脑 (多头) 的注意力不同,或者一个人大脑在不同时间下 (多头) 的注意力会增强或者减弱,甚至是波动。

        这里面涉及注意力分配问题,值得去考虑一下,是否对注意力进行分配更符合人类的思维方式呢？？？

        """

        self.d_model = d_model  # 词向量维度
        self.num_heads = num_heads  # 头的数量
        self.depth = d_model // num_heads  # 每个头的维度
        
        # 定义 Q, K, V 的线性变换
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 最终输出层
        self.W_o = nn.Linear(d_model, d_model)
        

    def split_heads(self, x, batch_size):

        """
        分割输入张量，使其适配多头注意力。
        
        形状变化：
        (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, depth)
        """

        # print(x.shape)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        print(x.shape)
        return x.permute(0, 2, 1, 3)  # 重新排列维度，以适应多头计算
    


    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        """
        计算缩放点积注意力。
        
        形状变化：
        - Q, K, V: (batch_size, num_heads, seq_len, depth)
        - 输出: (batch_size, num_heads, seq_len, depth)
        """

        d_k = Q.size(-1)  # 获取 K 的维度 depth
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # 计算注意力分数
        print(scores.shape, 1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 对填充部分进行掩码
        
        attention_weights = torch.softmax(scores, dim=-1)  # 计算注意力权重
        print(attention_weights.shape, 1)
        print(V.shape)
        output = torch.matmul(attention_weights, V)  # 加权求和
        print(output.shape)
        
        return output
    


    def forward(self, Q, K, V, mask=None):

        """
        前向传播。
        
        输入：
        - Q, K, V: (batch_size, seq_len, d_model)
        - mask: (batch_size, 1, 1, seq_len)，可选
        
        输出：
        - (batch_size, seq_len, d_model)
        """

        batch_size = Q.shape[0]
        
        # 通过线性层计算 Q, K, V
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        print(Q)
        
        # 分割为多个头
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        print(Q)
        print(Q.shape)
        
        # 计算注意力
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        print(attention_output.shape, 2)
        
        # 重新排列多头数据，使其合并回单一向量
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        
        # 变回 (batch_size, seq_len, d_model)
        attention_output = attention_output.view(batch_size, -1, self.d_model)
        print(attention_output.shape)
        
        # 通过最后的线性层
        output = self.W_o(attention_output)
        return output



if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 这里面因为没有进行词嵌入与位置编码，所以QKV进行随机初始化
    Q = torch.rand(batch_size, seq_len, d_model)
    K = torch.rand(batch_size, seq_len, d_model)
    V = torch.rand(batch_size, seq_len, d_model)
    
    output = mha(Q, K, V)
    print("输入Q: ", Q.shape)
    print("输入K: ", K.shape)
    print("输入V: ", V.shape)
    print("输出: ", output.shape)