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
from sklearn.metrics import r2_score

# ==========================================================================================
# 多头注意力
class WindowMultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super(WindowMultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.W_Q = nn.Linear(dim, dim, bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.fc = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(mh_attn_dropoutrate)
        self.fc_dropout = nn.Dropout(mh_fc_dropoutrate)

        self.relative_position_bias_table = nn.Parameter(torch.zeros(2 * window_size - 1, n_heads))
        coords_w = torch.arange(window_size)
        relative_position_index = coords_w[:, None] - coords_w[None, :]  # 2, Wh*Ww, Wh*Ww
        relative_position_index[:, :] += window_size - 1
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, seq_inputs, mask=None):
        # seq_inputs: [batch_size, seq_len, d_model]
        B, L, C = seq_inputs.shape
        d_k = C // self.n_heads
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: d_model
        # (B, S, D) -> (B, S, Head*d_k) -> (B, Head, S, d_k)
        # Q: [batch_size, n_heads, seq_len, d_k]
        Q = self.W_Q(seq_inputs).view(B, L, self.n_heads, -1).transpose(1, 2)
        # K: [batch_size, n_heads, seq_len, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(seq_inputs).view(B, L, self.n_heads, -1).transpose(1, 2)
        # V: [batch_size, n_heads, seq_len, d_v]
        V = self.W_V(seq_inputs).view(B, L, self.n_heads, -1).transpose(1, 2)

        # scores = Q*K.T : [batch_size, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(window_size, window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = scores + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW, N = mask.shape[0], mask.shape[1]
            attn = attn.view(-1, nW, self.n_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.n_heads, N, N)
            attn = nn.Softmax(dim=-1)(attn)
        else:
            attn = nn.Softmax(dim=-1)(attn)
        attn = self.attn_dropout(attn)
        context = torch.matmul(attn, V)

        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(B, L, -1)
        # output: [batch_size, seq_len, d_model]
        output = self.fc(context)
        output = self.fc_dropout(output)
        return output, attn


# ==========================================================================================
# 全连接层
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, dim):
        super(PoswiseFeedForwardNet, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=dim, out_channels=dim*4, kernel_size=1, stride=1, padding=0)
        self.Conv2 = nn.Conv1d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, stride=1, padding=1, groups=dim * 4)
        self.Conv3 = nn.Conv1d(in_channels=dim*4, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.GELU = nn.GELU()
        self.Dropout = nn.Dropout(ffn_dropoutrate)

    def forward(self, seq_inputs):
        # seq_inputs: [batch_size, seq_len, d_model]
        seq_inputs = seq_inputs.transpose(1, 2)
        output = self.Conv1(seq_inputs)
        output = self.GELU(output)
        output = self.Dropout(output)
        output = self.Conv2(output) + output
        output = self.GELU(output)
        output = self.Conv3(output)
        output = self.Dropout(output).transpose(1, 2)
        return output


# ==========================================================================================
# 池化层
class PatchMerging(nn.Module):
    def __init__(self, dim):
        super(PatchMerging, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.Conv1 = nn.Conv1d(in_channels=dim, out_channels=2 * dim, kernel_size=2, stride=2, padding=0)
        self.down_dropout = nn.Dropout(down_dropoutrate)

    def forward(self, seq_inputs):
        output = self.Conv1(seq_inputs.transpose(1, 2)).transpose(1, 2)
        output = self.down_dropout(output)
        return output

# ==========================================================================================
# Encoder
class EncoderLayer(nn.Module):
    def __init__(self, input_resolution, dim, shift_size, n_heads):
        super(EncoderLayer, self).__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.seq_attn = WindowMultiHeadAttention(dim=dim, n_heads=n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(dim=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(path_dropoutrate)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = input_resolution
            img_mask = torch.zeros((1, W))  # 1 H W 1
            w_slices = (slice(0, -window_size),
                        slice(-window_size, -shift_size),
                        slice(-shift_size, None))
            cnt = 0
            for w in w_slices:
                img_mask[:, w] = cnt
                cnt += 1
            mask_windows = img_mask.view(1, W // window_size, window_size)
            #print('mask_windows.shape=',mask_windows.shape)
            mask_windows = mask_windows.view(-1, window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, seq_inputs):
        B, _, C = seq_inputs.shape
        H, W = self.input_resolution
        # seq_input: [batch_size, src_len, d_model]
        residual = seq_inputs
        seq_inputs = self.norm1(seq_inputs)
        # seq_shift
        if self.shift_size > 0:
            shifted_x = torch.roll(seq_inputs, shifts=-self.shift_size, dims=1)
        else:
            shifted_x = seq_inputs
        #print('shifted_x.shape=', shifted_x.shape)
        # partition windows
        x_windows = shifted_x.view(B, W // self.window_size, self.window_size, C)
        #print('x_windows.shape=',x_windows.shape)
        x_windows = x_windows.contiguous().view(-1, self.window_size, C)
        #print('x_windows2.shape=', x_windows.shape)
        # W-MSA/SW-MSA
        attn_windows, attn = self.seq_attn(x_windows,  mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # merge windows
        shifted_x = attn_windows.view(B, -1, self.window_size, C)
        shifted_x = shifted_x.view(B, -1, C)
        # reverse cyclic shift
        if self.shift_size > 0:
            output = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            output = shifted_x
        output = self.dropout(output) + residual
        # FFN
        seq_outputs = self.norm2(output)
        seq_outputs = self.pos_ffn(seq_outputs)
        seq_outputs = self.dropout(seq_outputs) + output
        return seq_outputs, attn


class EncoderBlock(nn.Module):
    def __init__(self, input_resolution, downsample, dim, n_heads):
        super(EncoderBlock, self).__init__()
        self.encoder = EncoderLayer(input_resolution=input_resolution, dim=dim, shift_size=0, n_heads=n_heads)
        self.swinencoder = EncoderLayer(input_resolution=input_resolution, dim=dim, shift_size=shift_size, n_heads=n_heads)
        if downsample is True:
            self.downsample = PatchMerging(dim=dim)
        else:
            self.downsample = None

    def forward(self, seq_inputs):
        seq_outputs, attn = self.encoder(seq_inputs)
        seq_outputs, attn = self.swinencoder(seq_outputs)
        if self.downsample is not None:
            seq_outputs = self.downsample(seq_outputs)
        return seq_outputs, attn


class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads_list):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_features = int(d_model * 2 ** (self.num_layers - 1))
        self.seq_emb = nn.Conv1d(channels, d_model, kernel_size=5, stride=5)
        self.GELU = nn.GELU()
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(emb_dropoutrate)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EncoderBlock(input_resolution=(in_size[0], in_size[1] // (2 ** i_layer)),
                                 downsample=True if (i_layer < self.num_layers - 1) else False,
                                 dim=d_model * (2 ** i_layer),
                                 n_heads=num_heads_list[i_layer])
            self.layers.append(layer)
        self.norm2 = nn.LayerNorm(self.num_features)

        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, 40, d_model))
        trunc_normal_(self.absolute_pos_embed, std=.02)

    def forward(self, seq_inputs):
        seq_outputs = self.seq_emb(seq_inputs)
        ww6_1 = np.shape(seq_outputs)
        print('ww6_1=', ww6_1)
        seq_outputs = self.GELU(seq_outputs).transpose(1, 2)
        ww6_2 = np.shape(seq_outputs)
        print('ww6_2=', ww6_2)
        seq_outputs = self.norm1(seq_outputs)
        ww6_3 = np.shape(seq_outputs)
        print('ww6_3=', ww6_3)
        # seq_outputs = seq_outputs + self.absolute_pos_embed
        seq_outputs = self.dropout(seq_outputs)
        ww6_4 = np.shape(seq_outputs)
        print('ww6_4=', ww6_4)
        seq_attns = []
        for layer in self.layers:
            seq_outputs, attn = layer(seq_outputs)
            seq_attns.append(attn)
            ww6_5 = np.shape(seq_outputs)
            print('ww6_5=', ww6_5)
        seq_outputs = self.norm2(seq_outputs)
        ww6_6 = np.shape(seq_outputs)
        print('ww6_6=', ww6_6)
        return seq_outputs, seq_attns

class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.encoder = Encoder(num_layers=n_layers, num_heads_list=heads_list)
        self.Linear = nn.Linear(4800, 10)
    def forward(self, seq_inputs):
        seq_outputs, seq_attns = self.encoder(seq_inputs)
        seq_outputs = seq_outputs.view(seq_outputs.shape[0], -1)
        seq_outputs = self.Linear(seq_outputs)
        return seq_outputs
