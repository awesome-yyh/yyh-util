import torch
import torch.nn as nn


print("=== linear ===")
class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.b = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        x = x.mm(self.w)  # x.@(self.w)
        return x + self.b.expand_as(x)


print("=== MultiheadAttention ===")
m = nn.Linear(20, 30)
input = torch.randn(128, 20)
output = m(input)
print(output.size())

q = torch.tensor([[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [3, 4, 5]]], dtype=torch.float32)
k = torch.tensor([[[1, 2, 3], [3, 4, 5]],
                  [[4, 5, 6], [7, 8, 9]],
                  [[5, 4, 3], [2, 1, 0]]], dtype=torch.float32)
# v = torch.empty(k.shape, dtype=torch.float32)
v = k
print(q.shape, k.shape, v.shape)
token_attn = nn.MultiheadAttention(embed_dim=3, num_heads=1, dropout=0.3, batch_first=False)
attn_output, attn_output_weights = token_attn(q, k, v)
print(attn_output)
print(attn_output[0].shape)


print("=== Transformer ===")
# nn.Transformer主要由两部分构成：nn.TransformerEncoder和nn.TransformerDecoder。
# 而nn.TransformerEncoder又是由多个nn.TransformerEncoderLayer堆叠而成的
transformer = nn.Transformer(d_model=3, nhead=1, batch_first=False)
trans_output = transformer(q, k)
print(trans_output)
print(trans_output[-1])
print(trans_output[-1].shape)


print("=== TransformerEncoderLayer ===")
# nn.TransformerEncoderLayer 由自注意力和前向传播组成
encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
src = torch.rand(32, 10, 128)
out = encoder_layer(src)
print("TransformerEncoderLayer: ", out)
