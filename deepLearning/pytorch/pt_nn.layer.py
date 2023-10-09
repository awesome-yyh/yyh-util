import torch
import torch.nn as nn


line = nn.Linear(12 * 3, 7)
input1 = torch.randn(3, 12, requires_grad=True)
input1 = torch.hstack((input1, input1, input1))
print(input1, input1.shape)
print(line(input1))
print(input1.shape, line)