__author__ = 'minjinwu'

import math
import torch
import torch.nn as nn
import torch.functional as F


# 编码器类实现

## 词嵌入与位置编码
class Embedding(nn.Module):


    def __init__(self, vocab_size, embedding_dim, max_len=512, shared_weight=None):
        """
        词嵌入类，支持共享词嵌入和位置编码。
        
        参数:
        - vocab_size: 词汇表大小   总共有多少个词
        - embedding_dim: 词嵌入维度  每一个词转换为向量的维度大小
        - max_len: 句子最大长度（用于位置编码） 一次识别最多多少个词,少会padding,多会忽略
        - shared_weight: 可选，是否共享已有的嵌入层 (nn.Embedding)
        """
        super(Embedding, self).__init__()

        # 如果传入 shared_weight，则使用共享的嵌入层（就是encod和decoding进行权重共享）
        if shared_weight is not None:
            self.embedding = shared_weight  # 共享权重
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 位置编码
        self.positional_encoding = self.create_positional_encoding(max_len, embedding_dim)


    def create_positional_encoding(self, max_len, embedding_dim):
        """
        生成位置编码，采用 Transformer 的正弦余弦位置编码方法。
        """
        pos_enc = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        return pos_enc.unsqueeze(0)  # (1, max_len, embedding_dim) 方便 batch 处理


    def forward(self, input_ids):
        """
        前向传播：
        - 词索引 -> 词嵌入
        - 词嵌入 + 位置编码
        输入:
        - input_ids: (batch_size, seq_len) 形状的张量，表示词索引
        这里面的seq_len是你数据实际输入时候的长度,它可以比句子最大长度max_len要大,也可以小一些
        
        输出:
        - 嵌入后的张量，形状为 (batch_size, seq_len, embedding_dim)
        """
        embedded = self.embedding(input_ids)  # 词嵌入
        seq_len = input_ids.size(1)
        positional_enc = self.positional_encoding[:, :seq_len, :].to(embedded.device)  # 取前 seq_len 个位置编码

        return embedded + positional_enc  # 词嵌入 + 位置编码
    



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

        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        

    def split_heads(self, x, batch_size):

        """
        分割输入张量，使其适配多头注意力。
        
        形状变化：
        (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, depth)
        """

        # print(x.shape)
        x = x.view(batch_size, -1, self.num_heads, self.depth)

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
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 对填充部分进行掩码
        
        attention_weights = torch.softmax(scores, dim=-1)  # 计算注意力权重
    
        output = torch.matmul(attention_weights, V)  # 加权求和
        
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
   
        
        # 计算注意力
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        
        # 重新排列多头数据，使其合并回单一向量
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        

        # 变回 (batch_size, seq_len, d_model)
        attention_output = attention_output.view(batch_size, -1, self.d_model)


        # 通过最后的线性层
        output = self.W_o(attention_output)

        output = self.layer_norm(output + Q) # 进行层归一化与残差连接

        return output
    



## 前馈层（MLP：非线性激活+线性层）
class FeedForward(nn.Module):


    def __init__(self, d_model, d_ff, dropout=0.15):
    
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层线性变换（升维）
        self.relu = nn.ReLU()                # 非线性激活函数
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二层线性变换（降维）
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)   # 防止过拟合


    def forward(self, x):
        
        ff_out = self.fc2(self.dropout(self.relu(self.fc1(x))))
        ff_out  = self.layer_norm(ff_out + x)

        return ff_out




## 编码器层
class EncoderLayer(nn.Module):


    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        一个 Transformer 编码器层，包含多头注意力和前馈神经网络。
        
        参数:
        - d_model: 词向量的维度
        - num_heads: 多头注意力的头数
        - d_ff: 前馈网络的隐藏层维度
        - dropout: dropout 率
        """
        super(EncoderLayer, self).__init__()

        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)  # 用于多头注意力后的层归一化
        self.layer_norm2 = nn.LayerNorm(d_model)  # 用于前馈网络后的层归一化
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, mask=None):

        """
        前向传播：将输入经过多头注意力和前馈网络。
        
        输入:
        - x: (batch_size, seq_len, d_model)
        - mask: (batch_size, 1, 1, seq_len)，可选
        
        输出:
        - 编码器层的输出 (batch_size, seq_len, d_model)
        """
        
        # 多头注意力层
        attn_output = self.multihead_attention(x, x, x, mask) # 这里面x,x,x代表自注意力机制，即QKV都是通过x线性变换得到

        x = self.layer_norm1(attn_output + x)  # 残差连接 + 层归一化

        # 前馈网络
        ff_output = self.feed_forward(x)
        
        x = self.layer_norm2(ff_output + x)  # 残差连接 + 层归一化

        return x
    


## 多层编码器叠加
class Encoder(nn.Module):

    
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        """
        Transformer 编码器。
        
        参数:
        - vocab_size: 词汇表大小
        - embedding_dim: 词嵌入维度
        - num_layers: 编码器的层数（有多少个编码器堆叠）
        - num_heads: 多头注意力的头数
        - d_ff: 前馈网络的隐藏层维度
        - max_len: 句子的最大长度
        - dropout: dropout 率
        """
        super(Encoder, self).__init__()

        # 词嵌入和位置编码
        self.embedding = Embedding(vocab_size, embedding_dim, max_len)

        # 堆叠多个编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])


    def forward(self, x, mask=None):
        """
        前向传播：将输入通过多个编码器层。
        
        输入:
        - x: (batch_size, seq_len) 词索引
        - mask: (batch_size, 1, 1, seq_len)，可选
        
        输出:
        - 编码器的输出 (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)  # 词嵌入 + 位置编码
        
        for layer in self.layers:
            x = layer(x, mask)  # 通过每一层编码器
        
        return x





# 解码器实现
