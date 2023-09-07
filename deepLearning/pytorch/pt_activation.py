import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.style as style


style.use("bmh")
input = torch.arange(-3, 3.2, 0.2)


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


print("== Sigmoid ==")
# 1 / (1 + exp(-x)), 范围(0,1)
plt.plot(input, nn.Sigmoid()(input), label='Sigmoid')


print("== Tanh ==")
# (exp(x) - exp(-x)) / (exp(x) + exp(-x)), 范围(-1,1)
plt.plot(input, nn.Tanh()(input), label='Tanh')


print("== Softmax ==")
# exp(xi) / sum(exp(x)), 每个值的范围(0,1), 所有值之和为1
softmax = nn.Softmax(dim=1)(input.unsqueeze(0))
print("Softmax sum: ", torch.sum(softmax))
plt.plot(input, softmax.squeeze(), label='Softmax')

print("== LogSoftmax ==")
# log(exp(xi) / sum(exp(x)))
print(nn.LogSoftmax(dim=1)(input.unsqueeze(0)))


plt.legend()
plt.show()
