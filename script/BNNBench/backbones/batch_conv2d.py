import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math


class BatchConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        bias=True,
        ensemble_size=1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.ensemble_size = ensemble_size
        self.padding = padding
        self.stride = stride
        self.weight = nn.Parameter(
            torch.empty((out_channels, in_channels, kernel_size, kernel_size))
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_parameter(
            "R", nn.Parameter(torch.randn(ensemble_size, in_channels))
        )
        self.register_parameter(
            "S", nn.Parameter(torch.randn(ensemble_size, out_channels))
        )
        self.reset_parameters()

    def forward(self, x):
        batch_size = x.shape[0]
        examples_per_model = batch_size // self.ensemble_size
        R = torch.reshape(
            torch.tile(self.R, [1, examples_per_model]), [batch_size, self.in_channels]
        )
        S = torch.reshape(
            torch.tile(self.S, [1, examples_per_model]), [batch_size, self.out_channels]
        )
        R = torch.unsqueeze(R, -1)
        R = torch.unsqueeze(R, -1)
        S = torch.unsqueeze(S, -1)
        S = torch.unsqueeze(S, -1)
        xR = x * R
        out = F.conv2d(xR, self.weight, padding=self.padding, stride=self.stride, bias=self.bias)
        return out * S

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.R, 1.0, 0.2)
        nn.init.normal_(self.S, 1.0, 0.2)



class BatchConvTrans2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        bias=True,
        ensemble_size=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.ensemble_size = ensemble_size
        self.padding = padding
        self.stride = stride
        self.weight = nn.Parameter(
            torch.empty((in_channels, out_channels, kernel_size, kernel_size))
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_parameter(
            "R", nn.Parameter(torch.randn(ensemble_size, in_channels))
        )
        self.register_parameter(
            "S", nn.Parameter(torch.randn(ensemble_size, out_channels))
        )
        self.reset_parameters()

    def forward(self, x):
        batch_size = x.shape[0]
        examples_per_model = batch_size // self.ensemble_size
        R = torch.reshape(
            torch.tile(self.R, [1, examples_per_model]), [batch_size, self.in_channels]
        )
        S = torch.reshape(
            torch.tile(self.S, [1, examples_per_model]), [batch_size, self.out_channels]
        )
        R = torch.unsqueeze(R, -1)
        R = torch.unsqueeze(R, -1)
        S = torch.unsqueeze(S, -1)
        S = torch.unsqueeze(S, -1)
        xR = x * R
        out = F.conv_transpose2d(xR, self.weight, padding=self.padding, stride=self.stride, bias=self.bias)
        return out * S

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.R, 1.0, 0.2)
        nn.init.normal_(self.S, 1.0, 0.2)
