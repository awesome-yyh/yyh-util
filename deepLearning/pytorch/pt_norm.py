import torch
import torch.nn as nn


print("=== LayerNorm ===")
batch, sentence_length, embedding_dim = 20, 5, 10
embedding = torch.randn(batch, sentence_length, embedding_dim)
layer_norm = nn.LayerNorm(embedding_dim)
# Activate module
print(layer_norm(embedding))

print("=== DeepNorm ===")


print("=== RMSNorm ===")
# TODO RMSNorm
