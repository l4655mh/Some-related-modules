import math
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
import joblib



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
        # attn_mask = attn_mask.to(device)
        # # mask矩阵填充scores（用inf填充scores中与attn_mask中值为1位置相对应的元素）
        # scores.masked_fill_(attn_mask, -np.inf)
        # 对最后一个维度(v)做softmax
        attn = nn.Softmax(dim=-1)(scores)
        # scores*V: [batch_size, n_heads, seq_len, d_v]
        context = torch.matmul(attn, V)

        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        # output: [batch_size, seq_len, d_model]
        output = self.fc(context)
        return output, attn


# ==========================================================================================
# Encoder
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.seq_attn = MultiHeadAttention()

    def forward(self, seq_inputs):
        # seq_input: [batch_size, src_len, d_model]
        seq_outputs, attn = self.seq_attn(seq_inputs)
        return seq_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)  # Transformer中位置编码时固定的，不需要学习
        self.attn_layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, seq_inputs):
        # seq_inputs: [batch_size, seq_len, d_model]
        seq_outputs = self.pos_emb(seq_inputs)  # [batch_size, seq_len, d_model]
        seq_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等）
        for attn_layer in self.attn_layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出seq_outputs作为当前block的输入
            # seq_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            seq_outputs, seq_attn = attn_layer(seq_outputs)  # 传入的enc_outputs其实是input
            seq_attns.append(seq_attn)  # 这个只是为了可视化
        return seq_outputs, seq_attns

class block(nn.Module):
    def __init__(self,in_channels, out_channel_3_reduce, out_channel_3, out_channel_5_reduce,
                 out_channel_5, out_channel_7_reduce, out_channel_7):
        super(block, self).__init__()
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channel_3_reduce, kernel_size=1, padding=0),
            nn.ELU(),
            nn.Conv1d(in_channels=out_channel_3_reduce, out_channels=out_channel_3, kernel_size=3, padding=1),
            nn.ELU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channel_5_reduce, kernel_size=1, padding=0),
            nn.ELU(),
            nn.Conv1d(in_channels=out_channel_5_reduce, out_channels=out_channel_5, kernel_size=5, padding=2),
            nn.ELU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channel_7_reduce, kernel_size=1, padding=0),
            nn.ELU(),
            nn.Conv1d(in_channels=out_channel_7_reduce, out_channels=out_channel_7, kernel_size=7, padding=3),
            nn.ELU()
        )

    def forward(self, seq_inputs):
        # seq_input: [batch_size, src_len, d_model]
        seq_1 = self.conv3(seq_inputs)
        seq_2 = self.conv5(seq_inputs)
        seq_3 = self.conv7(seq_inputs)
        seq_outputs = torch.cat([seq_1, seq_2, seq_3], 1)
        return seq_outputs


class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.block1 = block(in_channels=8, out_channel_3_reduce=8, out_channel_3=32,
                             out_channel_5_reduce=8,out_channel_5=16, out_channel_7_reduce=8, out_channel_7=16)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.block2 = block(in_channels=64, out_channel_3_reduce=32, out_channel_3=96,
                             out_channel_5_reduce=32,out_channel_5=48, out_channel_7_reduce=32, out_channel_7=48)

    def forward(self, seq_inputs):
        # seq_input: [batch_size, src_len, d_model]
        seq_outputs = self.block1(seq_inputs)
        seq_outputs = self.pool(seq_outputs)
        seq_outputs = self.block2(seq_outputs)
        return seq_outputs


class CNN_Attention(nn.Module):
    def __init__(self):
        super(CNN_Attention, self).__init__()
        self.inception = Inception()
        self.encoder = Encoder()
        self.Linear = nn.Linear(576, 6).to(device)

    def forward(self, seq_inputs):
        # seq_inputs: [batch_size, d_model, seq_len]
        seq_outputs = self.inception(seq_inputs).transpose(1, 2)
        # seq_outputs: [batch_size, seq_len, d_model]
        seq_outputs, seq_attns = self.encoder(seq_outputs)
        seq_outputs = seq_outputs.view(seq_outputs.shape[0], -1)
        seq_outputs = self.Linear(seq_outputs)
        return seq_outputs




