import torch.nn as nn


class Linear(nn.Module):
    """搭建model"""
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)  # 包括两个可训练参数: weight和bias

    def forward(self, X):  # 直接调用对象时, 会自动将传入的参数传到forward函数当中进行计算
        logits = self.linear(X)

        return logits
