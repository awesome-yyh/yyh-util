import torch
import torch.nn as nn


input = torch.tensor([-2, -1, -0.5, 0, 0.5, 1, 2], dtype=torch.float32)

print("== relu ==")
# =max(0, x)
print(nn.ReLU()(input))

print("== LeakyReLU ==")
# >0是x, <0是negative_slope*x
print(nn.LeakyReLU(negative_slope=1e-2)(input))

print("== GELU ==")
# Gaussian Error Linear Units function
print(nn.GELU()(input))

print("== Sigmoid ==")
# 1 / (1 + exp(-x)), 范围(0,1)
print(nn.Sigmoid()(input))

print("== Tanh ==")
# (exp(x) - exp(-x)) / (exp(x) + exp(-x)), 范围(-1,1)
print(nn.Tanh()(input))

print("== Softmax ==")
# exp(xi) / sum(exp(x)), 每个值的范围(0,1), 所有值之和为1
print(nn.Softmax(dim=1)(input.unsqueeze(0)))
print(torch.sum(nn.Softmax(dim=1)(input.unsqueeze(0))))

print("== LogSoftmax ==")
# log(exp(xi) / sum(exp(x)))
print(nn.LogSoftmax(dim=1)(input.unsqueeze(0)))
