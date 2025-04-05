__author__ = 'minjinwu'

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads  # 每个头的维度

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
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 计算注意力
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 合并多头输出
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 输出线性层
        output = self.W_o(attn_output)

        # 残差连接 + 层归一化（注意 residual 是原始 Q）
        output = self.layer_norm(output + residual)

        return output


    

    ## 前馈层（MLP：非线性激活+线性层）
class FeedForward(nn.Module):


    def __init__(self, d_model, d_ff, dropout=0.1):
    
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层线性变换（升维）
        self.gelu = nn.GELU()                # 非线性激活函数
        self.fc2 = nn.Linear(d_ff, d_model)  # 第二层线性变换（降维）
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)   # 防止过拟合


    def forward(self, x):
        
        ff_out = self.fc2(self.dropout(self.gelu(self.fc1(x))))
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
        self.embedding = Embedding(vocab_size, embedding_dim, max_len, shared_weight= None)

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




    # 解码器层
class DecoderLayer(nn.Module):


    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        # 自注意力（带mask）
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        # 编码器-解码器注意力（交叉注意力）
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈层
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_output, tgt_mask=True, memory_mask=None):
        # 自注意力（目标语言）
        _x = self.self_attn(x, x, x, tgt_mask)  # 注意力使用目标语言自己与自己的注意力
        x = self.norm1(x + self.dropout(_x))  # 残差连接和归一化

        # 编码器-解码器注意力（源-目标交互）
        _x = self.cross_attn(x, enc_output, enc_output, memory_mask)  # 使用编码器输出和目标进行交互
        x = self.norm2(x + self.dropout(_x))  # 残差连接和归一化

        # 前馈网络
        _x = self.feed_forward(x)
        x = self.norm3(x + self.dropout(_x))  # 残差连接和归一化

        return x




    # 多层解码器的叠加
class Decoder(nn.Module):


    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1, shared_weight=None):
        super(Decoder, self).__init__()

        # 词嵌入层 + 位置编码
        self.embedding = Embedding(vocab_size, embedding_dim, max_len, shared_weight)
        
        # 解码器的每一层
        self.layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        # print(self.layers)

        # 输出层（将解码器的输出映射到词汇空间）
        # self.fc_out = nn.Linear(embedding_dim, vocab_size)

        # dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_output, tgt_mask=True, memory_mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(x)  # x 的形状为 (batch_size, seq_len)
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)  # 每一层解码器的输出

        # 通过输出层生成词汇概率分布
        print(x.shape)
        # x = self.fc_out(x)  # (batch_size, seq_len, vocab_size)
        
        return x

    


## 封装为tarnsformer
class Transformer(nn.Module):


    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(vocab_size, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)  # 输出层，映射到词汇表大小


    def forward(self, src, tgt, src_mask=None, tgt_mask=True):
        # 编码器输出
        enc_output = self.encoder(src, src_mask)
        
        # 解码器输出
        decoder_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        print(decoder_output.shape)
        
        # 通过全连接层进行输出映射
        output = self.fc_out(decoder_output.reshape(-1, decoder_output.shape[-1]))
        print(output.shape)

        return output


    def generate_tgt_mask(self, seq_len):
        # 生成目标序列的mask（防止看见未来的词）
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool()  # 下三角矩阵
        return mask.unsqueeze(0).unsqueeze(0)  # 扩展到 (1, 1, seq_len, seq_len) 的形状


    def generate_src_mask(self, seq_len):
        # 生成源序列的mask（如果需要）
        return torch.ones(1, 1, seq_len, seq_len).to(torch.bool)

# 训练过程
def train_transformer(model, train_data, vocab_size, num_epochs=10, lr=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_data:  # 假设train_data是一个可迭代的数据集
            optimizer.zero_grad()

            # 准备掩码
            src_mask = model.generate_src_mask(src.size(1))
            tgt_mask = model.generate_tgt_mask(tgt.size(1))

            # 前向传播
            output = model(src, tgt, src_mask, tgt_mask)

            # 计算损失
            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_data)}")


if __name__ == '__main__':

    vocab_size = 10000  # 词汇表大小
    embedding_dim = 512  # 词嵌入维度
    num_layers = 6  # 编码器和解码器的层数
    num_heads = 8  # 多头注意力的头数
    d_ff = 2048  # 前馈网络的隐藏层维度
    max_len = 512  # 句子最大长度
    dropout = 0.1  # dropout比率

    # 创建 Transformer 模型
    model = Transformer(vocab_size, embedding_dim, num_layers, num_heads, d_ff, max_len, dropout)

    # 生成一些测试数据
    batch_size = 32  # 批量大小
    src_len = 50  # 源序列长度
    tgt_len = 60  # 目标序列长度

    # 随机生成源序列和目标序列（模拟训练时的输入数据）
    src = torch.randint(0, vocab_size, (batch_size, src_len))  # (batch_size, src_len)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))  # (batch_size, tgt_len)

    # 生成源序列和目标序列的mask（防止模型看到未来的词）
    src_mask = torch.ones(batch_size, 1, 1, src_len).to(torch.bool)
    tgt_mask = model.generate_tgt_mask(tgt_len).to(torch.bool)

    # 前向传播
    output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)