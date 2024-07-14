'''
Author: yyh owyangyahe@126.com
Date: 2023-08-22 15:44:23
LastEditors: yyh owyangyahe@126.com
LastEditTime: 2024-02-18 08:14:30
FilePath: /mypython/yyh-util/deepLearning/pytorch/pt_activation.py
Description: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("darkgrid")
input = torch.arange(-5, 5.0, 0.2)

print("== Sigmoid ==")
# 1 / (1 + exp(-x)), 范围(0,1)
sns.lineplot(x=input, y=nn.Sigmoid()(input), linestyle='-', label='Sigmoid')

print("== Tanh ==")
# (exp(x) - exp(-x)) / (exp(x) + exp(-x)), 范围(-1,1)
sns.lineplot(x=input, y=nn.Tanh()(input), linestyle='--', label='Tanh')

print("== relu ==")
# relu = max(x, 0)
sns.lineplot(x=input, y=nn.ReLU()(input), linestyle=':', label='ReLU')

print("== LeakyReLU ==")
# >0是x, <0是negative_slope*x
sns.lineplot(x=input, y=nn.LeakyReLU(negative_slope=1e-2)(input), linestyle='-.', label='LeakyReLU')


print("== GELU ==")
# Gaussian Error Linear Units function
# GELU = x * f(x) = x * sigmoid(1.702*x)
sns.lineplot(x=input, y=nn.GELU()(input), linestyle='-', label='GELU')


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


print("== QuickGELU ==")
sns.lineplot(x=input, y=QuickGELU()(input), linestyle='--', label='QuickGELU')

print("== Softmax ==")
# exp(xi) / sum(exp(x)), 每个值的范围(0,1), 所有值之和为1
softmax = nn.Softmax(dim=1)(input.unsqueeze(0))
sns.lineplot(x=input, y=softmax.squeeze(), linestyle=':', label='Softmax')
print("Softmax sum: ", torch.sum(softmax))

print("== LogSoftmax ==")
# log(exp(xi) / sum(exp(x)))
sns.lineplot(x=input, y=nn.LogSoftmax(dim=1)(input.unsqueeze(0)).squeeze(), linestyle='-.', label='LogSoftmax')

print("== Hardswish ==")
# <-3是0，>3是x，其他是x*(x+3)/6
sns.lineplot(x=input, y=nn.Hardswish()(input), linestyle='-.', label='Hardswish')


print("== GLU ==")
# gated linear unit
sns.lineplot(nn.GLU()(input), label='GLU')


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor):
        x = torch.chunk(x, 2, dim=-1)
        return F.gelu(x[0]) * x[1]


print("== GeGLU ==")
sns.lineplot(GEGLU()(input), label='GEGLU')


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor):
        x = torch.chunk(x, 2, dim=-1)
        return F.silu(x[0]) * x[1]


print("== SwiGLU ==")
sns.lineplot(SwiGLU()(input), label='SwiGLU')

plt.show()
