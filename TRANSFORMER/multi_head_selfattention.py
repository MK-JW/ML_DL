__author__ = 'minjinwu'


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

## 多头注意力机制（可学习动态维度分配）
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicMultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, min_dim=16, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.min_dim = min_dim

        # 可学习维度权重（用于 sigmoid gate）
        self.head_weights = nn.Parameter(torch.randn(num_heads))

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # 注册用于软掩码的位置向量
        self.register_buffer("pos_range", torch.arange(d_model).float())

    def _create_soft_mask(self, start, dim):
        positions = self.pos_range.to(start.device)
        left = torch.sigmoid((positions - start) * 10)
        right = torch.sigmoid((start + dim - positions) * 10)
        return left * right  # shape: [d_model]

    def forward(self, query, key_padding_mask=None, return_aux_loss=False):
        B, T, _ = query.shape

        # === 动态维度分配（可导）===
        gate = F.softmax(self.head_weights)  # [num_heads]
        adjustable_dim = self.d_model - self.min_dim * self.num_heads
        head_dims = self.min_dim + gate * adjustable_dim  # 保证总和为 d_model
        head_dims = F.relu(head_dims)

        # === QKV projection ===
        q_all, k_all, v_all = self.qkv_proj(query).chunk(3, dim=-1)

        # === 多头注意力按维度切片（软掩码）===
        outputs = []
        start = torch.tensor(0.0, device=query.device)
        for i in range(self.num_heads):
            curr_dim = head_dims[i]
            mask = self._create_soft_mask(start, curr_dim).view(1, 1, -1)
            q = q_all * mask
            k = k_all * mask
            v = v_all * mask

            scores = torch.matmul(q, k.transpose(-2, -1)) / (curr_dim + 1e-6).sqrt()
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1), -1e9)
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            outputs.append(torch.matmul(attn, v))

            start += curr_dim

        # === 输出整合 ===
        output = self.out_proj(sum(outputs))  # [B, T, d_model]
        output = self.layer_norm(output + query)

        if return_aux_loss:
            min_loss = F.relu(self.min_dim - head_dims).sum()
            total_loss = (head_dims.sum() - self.d_model) ** 2
            aux_loss = total_loss + 0.1 * min_loss
            return output, aux_loss
        return output

    

mha = DynamicMultiHeadAttention(d_model=64, num_heads=4, min_dim=8)
x = torch.randn(2, 10, 64)  # B, T, d_model

optimizer = torch.optim.Adam(mha.parameters(), lr=1e-3)

for step in range(10):
    out, aux_loss = mha(x, return_aux_loss=True)
    task_loss = out.mean()  # 假设一个主损失
    loss = task_loss + aux_loss  # 总损失
    loss.backward()

    print(f"[{step}] task_loss={task_loss.item():.4f}  aux_loss={aux_loss.item():.4f}")
    print("head_dims =", mha.min_dim + torch.sigmoid(mha.head_weights) * (mha.d_model - mha.min_dim * mha.num_heads))
    print("Grad on head_weight_logits:", mha.head_weights.grad)

    optimizer.step()
    optimizer.zero_grad()
