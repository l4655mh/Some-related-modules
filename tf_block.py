import math
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from timm.models.layers import trunc_normal_
from scipy.interpolate import make_interp_spline
from torch.nn.utils import weight_norm
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score

# ==========================================================================================
# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pe.require_grad = False
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==========================================================================================
# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.attn_dropout = nn.Dropout(mh_attn_dropoutrate)
        self.fc_dropout = nn.Dropout(mh_fc_dropoutrate)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, seq_inputs):
        # seq_inputs: [batch_size, seq_len, d_model]
        residual, batch_size = seq_inputs, seq_inputs.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: d_model
        # (B, S, D) -> (B, S, Head*d_k) -> (B, Head, S, d_k)
        # Q: [batch_size, n_heads, seq_len, d_k]
        Q = self.W_Q(seq_inputs).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, seq_len, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(seq_inputs).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, seq_len, d_v]
        V = self.W_V(seq_inputs).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # scores = Q*K.T : [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        # # 生成一个上三角掩码矩阵
        # attn_mask = torch.triu(torch.ones(scores.shape), diagonal=1).type(torch.uint8)
        # # mask矩阵填充scores（用inf填充scores中与attn_mask中值为1位置相对应的元素）
        # scores.masked_fill_(attn_mask, -np.inf)
        # 对最后一个维度(v)做softmax
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.attn_dropout(attn)
        # scores*V: [batch_size, n_heads, seq_len, d_v]
        context = torch.matmul(attn, V)

        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        # output: [batch_size, seq_len, d_model]
        output = self.fc(context)
        output = self.fc_dropout(output)
        output = self.norm1(output + residual)
        return output, attn


# ==========================================================================================
# 全连接层
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(ffn_dropoutrate),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(ffn_dropoutrate)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, seq_inputs):
        # seq_inputs: [batch_size, seq_len, d_model]
        residual = seq_inputs
        seq_outputs = self.fc(seq_inputs)
        output = self.norm(seq_outputs + residual)# [batch_size, seq_len, d_model]
        return output


# ==========================================================================================
# Encoder
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.seq_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, seq_inputs):
        # seq_input: [batch_size, src_len, d_model]
        seq_outputs, attn = self.seq_attn(seq_inputs)
        seq_outputs = self.pos_ffn(seq_outputs)
        return seq_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.seq_emb = nn.Conv1d(in_channels=channel, out_channels=d_model, kernel_size=3, padding=1)  # token Embedding
        self.pos_emb = PositionalEncoding(d_model)  # Transformer中位置编码时固定的，不需要学习
        self.attn_layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, seq_inputs):
        # seq_inputs: [batch_size, seq_len, d_model]
        seq_outputs = self.seq_emb(seq_inputs).transpose(1, 2)  # [batch_size, seq_len, d_model]
        seq_outputs = self.pos_emb(seq_outputs)  # [batch_size, seq_len, d_model]
        seq_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等）
        for attn_layer in self.attn_layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出seq_outputs作为当前block的输入
            # seq_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            seq_outputs, seq_attn = attn_layer(seq_outputs)  # 传入的enc_outputs其实是input
            seq_attns.append(seq_attn)  # 这个只是为了可视化
        return seq_outputs, seq_attns


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.Linner1 = nn.Linear(128 * 200, 10)

    def forward(self, seq_inputs):
        # seq_outputs: [batch_size, seq_len, d_model]
        seq_outputs, seq_attns = self.encoder(seq_inputs)
        output = seq_outputs.view(seq_outputs.shape[0], -1)
        output = self.Linner1(output)
        return output

