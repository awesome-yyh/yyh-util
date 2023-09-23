import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.style as style


style.use("bmh")
input = torch.arange(-5, 5.2, 0.2)

print("== Sigmoid ==")
# 1 / (1 + exp(-x)), 范围(0,1)
plt.plot(input, nn.Sigmoid()(input), label='Sigmoid')


print("== Tanh ==")
# (exp(x) - exp(-x)) / (exp(x) + exp(-x)), 范围(-1,1)
plt.plot(input, nn.Tanh()(input), label='Tanh')


print("== relu ==")
# relu = max(x, 0)
plt.plot(input, nn.ReLU()(input), label='relu')


print("== LeakyReLU ==")
# >0是x, <0是negative_slope*x
plt.plot(input, nn.LeakyReLU(negative_slope=1e-2)(input), label='LeakyReLU')


print("== GELU ==")
# Gaussian Error Linear Units function
# GELU = x * f(x)
plt.plot(input, nn.GELU()(input), label='GELU')


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


print("== QuickGELU ==")
plt.plot(input, QuickGELU()(input), label='QuickGELU')


print("== Softmax ==")
# exp(xi) / sum(exp(x)), 每个值的范围(0,1), 所有值之和为1
softmax = nn.Softmax(dim=1)(input.unsqueeze(0))
print("Softmax sum: ", torch.sum(softmax))
plt.plot(input, softmax.squeeze(), label='Softmax')

print("== LogSoftmax ==")
# log(exp(xi) / sum(exp(x)))
print(nn.LogSoftmax(dim=1)(input.unsqueeze(0)))

print("== Hardswish ==")
# <-3是0，>3是x，其他是x*(x+3)/6
plt.plot(input, nn.Hardswish()(input), label='Hardswish')


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


plt.legend()
plt.show()
