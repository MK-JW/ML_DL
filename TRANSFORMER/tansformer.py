__author__ = 'minjinwu'

# 导入所需要的包
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置超参数

# 导入数据

# 模型的构建


## 词嵌入与位置编码
class Embedding(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, max_len=512, shared_weight=None):
        """
        词嵌入类，支持共享词嵌入和位置编码。
        
        参数:
        - vocab_size: 词汇表大小   每个词对应的索引
        - embedding_dim: 词嵌入维度  每一个词转换为向量的维度大小
        - max_len: 句子最大长度（用于位置编码） 一次识别最多多少个词
        - shared_weight: 可选，是否共享已有的嵌入层 (nn.Embedding)
        """
        super(Embedding, self).__init__()

        # 如果传入 shared_weight，则使用共享的嵌入层
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
        
        输出:
        - 嵌入后的张量，形状为 (batch_size, seq_len, embedding_dim)
        """
        embedded = self.embedding(input_ids)  # 词嵌入
        seq_len = input_ids.size(1)
        positional_enc = self.positional_encoding[:, :seq_len, :].to(embedded.device)  # 取前 seq_len 个位置编码

        return embedded + positional_enc  # 词嵌入 + 位置编码
    
# 设置参数
vocab_size = 100  # 假设词汇表大小为100
embedding_dim = 16  # 词嵌入维度为16
max_len = 10  # 句子最大长度

# 创建词嵌入模型
embedding_layer = Embedding(vocab_size, embedding_dim, max_len)

# 生成测试输入（batch_size=2, seq_len=5），随机选择词索引
input_ids = torch.randint(0, vocab_size, (2, 5))
print("输入词索引:")
print(input_ids)

# 前向传播
output = embedding_layer(input_ids)
print("\n输出词嵌入+位置编码:")
print(output)


## 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        多头注意力机制。
        
        参数：
        - d_model: 词向量的维度（例如 512)S
        - num_heads: 头的数量（例如 8)
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
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
        return output

if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    Q = torch.rand(batch_size, seq_len, d_model)
    K = torch.rand(batch_size, seq_len, d_model)
    V = torch.rand(batch_size, seq_len, d_model)
    
    output = mha(Q, K, V)
    print("输入Q: ", Q.shape)
    print("输入K: ", K.shape)
    print("输入V: ", V.shape)
    print("输出: ", output.shape)



# ## 编码器

# class PositionalEncoding(nn.Module):
#     def __init__(self, embedding_dim, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, embedding_dim)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:, :x.size(1)]

# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, embedding_dim, num_heads):
#         super(MultiHeadSelfAttention, self).__init__()
#         assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
#         self.num_heads = num_heads
#         self.head_dim = embedding_dim // num_heads
#         self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
#         self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
#     def forward(self, x, mask=None):
#         batch_size, seq_length, embedding_dim = x.shape
#         qkv = self.qkv_proj(x).reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
#         q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#         attention = F.softmax(scores, dim=-1)
#         output = torch.matmul(attention, v)
#         output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embedding_dim)
#         return self.out_proj(output)

# class FeedForward(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim):
#         super(FeedForward, self).__init__()
#         self.fc1 = nn.Linear(embedding_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, embedding_dim)
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x):
#         return self.fc2(self.dropout(F.relu(self.fc1(x))))

# class TransformerEncoderLayer(nn.Module):
#     def __init__(self, embedding_dim, num_heads, hidden_dim):
#         super(TransformerEncoderLayer, self).__init__()
#         self.attention = MultiHeadSelfAttention(embedding_dim, num_heads)
#         self.norm1 = nn.LayerNorm(embedding_dim)
#         self.ffn = FeedForward(embedding_dim, hidden_dim)
#         self.norm2 = nn.LayerNorm(embedding_dim)
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x, mask=None):
#         attn_output = self.attention(x, mask)
#         x = self.norm1(x + self.dropout(attn_output))
#         ffn_output = self.ffn(x)
#         x = self.norm2(x + self.dropout(ffn_output))
#         return x

# class TransformerEncoder(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_len=5000):
#         super(TransformerEncoder, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.pos_encoding = PositionalEncoding(embedding_dim, max_len)
#         self.layers = nn.ModuleList([
#             TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)
#         ])
        
#     def forward(self, x, mask=None):
#         x = self.embedding(x)
#         x = self.pos_encoding(x)
#         for layer in self.layers:
#             x = layer(x, mask)
#         return x
    

# ## 解码器
# class MultiHeadSelfAttentionDecoder(nn.Module):
#     def __init__(self, embedding_dim, num_heads):
#         super(MultiHeadSelfAttentionDecoder, self).__init__()
#         assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
#         self.num_heads = num_heads
#         self.head_dim = embedding_dim // num_heads
#         self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
#         self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
#     def forward(self, x, mask=None):
#         batch_size, seq_length, embedding_dim = x.shape
#         qkv = self.qkv_proj(x).reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
#         q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#         attention = F.softmax(scores, dim=-1)
#         output = torch.matmul(attention, v)
#         output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_length, embedding_dim)
#         return self.out_proj(output)

# class TransformerDecoderLayer(nn.Module):
#     def __init__(self, embedding_dim, num_heads, hidden_dim):
#         super(TransformerDecoderLayer, self).__init__()
#         self.self_attention = MultiHeadSelfAttentionDecoder(embedding_dim, num_heads)
#         self.encoder_decoder_attention = MultiHeadSelfAttention(embedding_dim, num_heads)
#         self.norm1 = nn.LayerNorm(embedding_dim)
#         self.norm2 = nn.LayerNorm(embedding_dim)
#         self.ffn = FeedForward(embedding_dim, hidden_dim)
#         self.norm3 = nn.LayerNorm(embedding_dim)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
#         # Self-attention in decoder
#         self_attn_output = self.self_attention(x, tgt_mask)
#         x = self.norm1(x + self.dropout(self_attn_output))

#         # Encoder-decoder attention
#         encoder_decoder_attn_output = self.encoder_decoder_attention(x, src_mask)
#         x = self.norm2(x + self.dropout(encoder_decoder_attn_output))

#         # Feed Forward Network
#         ffn_output = self.ffn(x)
#         x = self.norm3(x + self.dropout(ffn_output))

#         return x

# class TransformerDecoder(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_len=5000):
#         super(TransformerDecoder, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.pos_encoding = PositionalEncoding(embedding_dim, max_len)
#         self.layers = nn.ModuleList([
#             TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)
#         ])
        
#     def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
#         x = self.embedding(x)
#         x = self.pos_encoding(x)
#         for layer in self.layers:
#             x = layer(x, encoder_output, src_mask, tgt_mask)
#         return x





# ## 完整的transformer代码

# # 位置编码 (Positional Encoding)
# class PositionalEncoding(nn.Module):
#     def __init__(self, embedding_dim, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, embedding_dim)
#         for pos in range(max_len):
#             for i in range(0, embedding_dim, 2):
#                 pe[pos, i] = math.sin(pos / (10000 ** (i / embedding_dim)))
#                 pe[pos, i + 1] = math.cos(pos / (10000 ** (i / embedding_dim)))
#         pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, embedding_dim]
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]

# # 多头自注意力机制
# class MultiHeadAttention(nn.Module):
#     def __init__(self, embedding_dim, num_heads):
#         super(MultiHeadAttention, self).__init__()
#         assert embedding_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
#         self.num_heads = num_heads
#         self.head_dim = embedding_dim // num_heads
#         self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)
#         self.out_proj = nn.Linear(embedding_dim, embedding_dim)

#     def forward(self, x, mask=None):
#         batch_size, seq_len, embedding_dim = x.shape
#         qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
#         q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
#         scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, float('-inf'))
#         attention = F.softmax(scores, dim=-1)
#         output = torch.matmul(attention, v)
#         output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embedding_dim)
#         return self.out_proj(output)

# # 前馈神经网络（Feed-Forward Network）
# class FeedForward(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim):
#         super(FeedForward, self).__init__()
#         self.fc1 = nn.Linear(embedding_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, embedding_dim)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         return self.fc2(self.dropout(F.relu(self.fc1(x))))

# # 编码器层（Encoder Layer）
# class EncoderLayer(nn.Module):
#     def __init__(self, embedding_dim, num_heads, hidden_dim):
#         super(EncoderLayer, self).__init__()
#         self.self_attention = MultiHeadAttention(embedding_dim, num_heads)
#         self.norm1 = nn.LayerNorm(embedding_dim)
#         self.ffn = FeedForward(embedding_dim, hidden_dim)
#         self.norm2 = nn.LayerNorm(embedding_dim)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x, mask=None):
#         # 自注意力层
#         attn_output = self.self_attention(x, mask)
#         x = self.norm1(x + self.dropout(attn_output))

#         # 前馈神经网络层
#         ffn_output = self.ffn(x)
#         x = self.norm2(x + self.dropout(ffn_output))

#         return x

# # 解码器层（Decoder Layer）
# class DecoderLayer(nn.Module):
#     def __init__(self, embedding_dim, num_heads, hidden_dim):
#         super(DecoderLayer, self).__init__()
#         self.self_attention = MultiHeadAttention(embedding_dim, num_heads)
#         self.encoder_attention = MultiHeadAttention(embedding_dim, num_heads)
#         self.norm1 = nn.LayerNorm(embedding_dim)
#         self.norm2 = nn.LayerNorm(embedding_dim)
#         self.ffn = FeedForward(embedding_dim, hidden_dim)
#         self.norm3 = nn.LayerNorm(embedding_dim)
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
#         # 自注意力层
#         self_attn_output = self.self_attention(x, tgt_mask)
#         x = self.norm1(x + self.dropout(self_attn_output))

#         # 编码器-解码器注意力层
#         encoder_attn_output = self.encoder_attention(x, encoder_output)
#         x = self.norm2(x + self.dropout(encoder_attn_output))

#         # 前馈神经网络层
#         ffn_output = self.ffn(x)
#         x = self.norm3(x + self.dropout(ffn_output))

#         return x

# # Transformer 编码器（Encoder）
# class TransformerEncoder(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_len=5000):
#         super(TransformerEncoder, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.pos_encoding = PositionalEncoding(embedding_dim, max_len)
#         self.layers = nn.ModuleList([
#             EncoderLayer(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)
#         ])

#     def forward(self, x, mask=None):
#         x = self.embedding(x)
#         x = self.pos_encoding(x)
#         for layer in self.layers:
#             x = layer(x, mask)
#         return x

# # Transformer 解码器（Decoder）
# class TransformerDecoder(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_len=5000):
#         super(TransformerDecoder, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.pos_encoding = PositionalEncoding(embedding_dim, max_len)
#         self.layers = nn.ModuleList([
#             DecoderLayer(embedding_dim, num_heads, hidden_dim) for _ in range(num_layers)
#         ])

#     def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
#         x = self.embedding(x)
#         x = self.pos_encoding(x)
#         for layer in self.layers:
#             x = layer(x, encoder_output, src_mask, tgt_mask)
#         return x

# # 完整的 Transformer 模型（Encoder + Decoder）
# class Transformer(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_len=5000):
#         super(Transformer, self).__init__()
#         self.encoder = TransformerEncoder(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_len)
#         self.decoder = TransformerDecoder(vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_len)
#         self.fc_out = nn.Linear(embedding_dim, vocab_size)

#     def forward(self, src, tgt, src_mask=None, tgt_mask=None):
#         encoder_output = self.encoder(src, src_mask)
#         decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
#         return self.fc_out(decoder_output)

# # 创建模型实例
# model = Transformer(vocab_size=10000, embedding_dim=512, num_heads=8, hidden_dim=2048, num_layers=6)

# # 输入示例
# src = torch.randint(0, 10000, (32, 20))  # 假设一个 batch_size=32，句子长度=20的源序列
# tgt = torch.randint(0, 10000, (32, 20))  # 目标序列

# # 前向传播
# output = model(src, tgt)


# 模型的训练

# 当前文件夹的测试

if __name__ == "__main__":
    vocab_size = 10000
    embedding_dim = 512
    max_len = 100

    embedding_layer = Embedding(vocab_size, embedding_dim, max_len)

    input_sentences = torch.tensor([
        [1, 23, 456, 7890, 5],   
        [34, 5678, 90, 1234, 6]  
    ])

    output_embeddings = embedding_layer(input_sentences)
    print("嵌入后的输出形状:", output_embeddings.shape)  # 应该是 (2, 5, 512)