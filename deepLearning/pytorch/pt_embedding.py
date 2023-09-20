import torch
import torch.nn as nn

# an Embedding module containing 10 tensors of size 3
embedding = nn.Embedding(10, 3)
# a batch of 2 samples of 4 indices each
input = torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]])
embedding(input)


print("=== Embedding ===")
embedding = nn.Embedding(10, 3, padding_idx=0)
input = torch.LongTensor([[0, 2, 0, 5]])
embedding(input)

# TODO Fixed Absolute, Learned Absolute, Fixed Relative, Learned Relative, RoPE, ALiBi
