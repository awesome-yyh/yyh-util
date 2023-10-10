import math
import torch
import torch.nn as nn


torch.random.manual_seed(42)

print("=== linear ===")
input1 = torch.randn(3, 4, requires_grad=True)
linear = nn.Linear(4, 7)
print(linear(input1))
print(input1.shape, linear)


class Linear(nn.Module):
    """组合搭建model"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)  # 包括两个可训练参数: weight和bias

    def forward(self, X):  # 直接调用对象时, 会自动将参数传入forward进行计算
        logits = self.linear(X)

        return logits


linear_module = Linear(4, 7)
print(linear_module(input1))
print(input1.shape, linear_module)


class MyLinear(nn.Module):
    """实现Linear"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = nn.Parameter(torch.empty((in_features, out_features)))
        self.b = nn.Parameter(torch.empty(out_features))
        
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
    
    def forward(self, x):
        x = x.matmul(self.w)  # x @ self.w
        return x + self.b.expand_as(x)


print(MyLinear(4, 7)(input1))
print(input1.shape, Linear(4, 7))


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
