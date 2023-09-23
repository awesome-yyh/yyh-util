import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("darkgrid")
input = torch.arange(-5, 5.2, 0.2)

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
# GELU = x * f(x)
sns.lineplot(x=input, y=nn.GELU()(input), linestyle='-', label='GELU')


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


print("== QuickGELU ==")
sns.lineplot(x=input, y=QuickGELU()(input), linestyle='--', label='QuickGELU')

print("== Softmax ==")
# exp(xi) / sum(exp(x)), 每个值的范围(0,1), 所有值之和为1
softmax = nn.Softmax(dim=1)(input.unsqueeze(0))
print("Softmax sum: ", torch.sum(softmax))
sns.lineplot(x=input, y=softmax.squeeze(), linestyle=':', label='Softmax')

print("== LogSoftmax ==")
# log(exp(xi) / sum(exp(x)))
# sns.lineplot(x=input, y=nn.LogSoftmax(dim=1)(input.unsqueeze(0)).squeeze(), linestyle='-.', label='LogSoftmax')

print("== Hardswish ==")
# <-3是0，>3是x，其他是x*(x+3)/6
sns.lineplot(x=input, y=nn.Hardswish()(input), linestyle='-.', label='Hardswish')

class _GLUBaseModule(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x):
        # dim=-1 breaks in jit for pt<1.10
        x1, x2 = x.chunk(2, dim=(x.ndim - 1))
        return x1 * self.activation_fn(x2)


print("== GLU ==")
print(nn.GLU()(torch.randn(4, 2)))
# plt.plot(input, _GLUBaseModule()(input), label='GLU')


class GEGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.gelu)


print("== GeGLU ==")
# plt.plot(input, GEGLU()(input), label='GEGLU')


class SwiGLU(_GLUBaseModule):
    def __init__(self):
        super().__init__(F.silu)


print("== SwiGLU ==")
# plt.plot(input, SwiGLU()(input), label='SwiGLU')

plt.show()
