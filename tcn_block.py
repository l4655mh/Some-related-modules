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



class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConstantPad1d((padding, 0), 0),
            weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((padding, 0), 0),
            weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=0, dilation=dilation)),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        if n_inputs != n_outputs:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
        else:
            self.downsample = None
        self.relu = nn.ReLU()

    def forward(self, seq_inputs):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        residual = seq_inputs
        output = self.conv1(seq_inputs)
        output = self.conv2(output)
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(output + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        self.num_layers = len(num_channels)
        self.network = nn.ModuleList()
        for i in range(self.num_layers):
            dilation = 2 ** i  # 膨胀系数：1，2，4，8……
            padding = (kernel_size - 1) * dilation  # 填充系数
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 每一层的输入通道数
            out_channels = num_channels[i]  # 每一层的输出通道数
            layer = TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation, padding=padding,
                                  dropout=dropout)
            self.network.append(layer)

    def forward(self, seq_inputs):
        output = seq_inputs
        for layer in self.network:
            output = layer(output)
        return output


class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=12, num_channels=[32, 64, 64, 32, 10], kernel_size=31, dropout=0.2)
        self.Linner1 = nn.Linear(10*200, 256)
        self.Linner2 = nn.Linear(256, 10)

    def forward(self, seq_inputs):
        output = self.tcn(seq_inputs)
        output = output.view(output.shape[0], -1)
        output = self.Linner1(output)
        output = self.Linner2(output)
        return output

