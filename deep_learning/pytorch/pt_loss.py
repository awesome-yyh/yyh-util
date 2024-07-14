#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from focal_loss import FocalLoss


torch.random.manual_seed(3)

print("=== MSELoss ===")
# 输出和Label的均方差
# loss = mean((y-h)^2)
criterion = nn.MSELoss()

target = torch.tensor([0.1, 0.3, 0.5])
output = torch.tensor([0.1, 0.3, 0.5])
print("loss应该小: ", criterion(output, target))
output = torch.tensor([1.1, 1.3, 1.5])
print("loss应该大: ", criterion(output, target))


print("=== BCEWithLogitsLoss ===")
# 使用sigmoid，适用于2分类、多标签分类
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 1.0, 1.0]))  # pos_weight: 每个类别的正例的权重, 通常设为设为负样本数/正样本数

target = torch.tensor([0.0, 1, 1])
output = torch.tensor([-10.0, 10, 10])
print("loss应该小: ", criterion(output, target))
output = torch.tensor([10.0, -10, -10])
print("loss应该大: ", criterion(output, target))


print("=== CrossEntropyLoss ===")
# 使用softmax，适用于多分类
criterion = nn.CrossEntropyLoss()

target = torch.tensor([0.0, 1, 2])
output = torch.tensor([-10.0, 0, 10])
print("loss应该小: ", criterion(output, target))
output = torch.tensor([-10.0, 10, -10])
print("loss应该大: ", criterion(output, target))

output = torch.tensor([[-1.0, 0, 120, 1.2], [1.0, -10, 1.0, 1.2], [9.0, 0, -10, 9.1]])
target = torch.tensor([-100, 1, 2])
print("loss: ", criterion(output, target))

# output = torch.tensor([[-1.0, 0, 10, 1.2], [-1.0, 0, 10, 2.2], [9.0, 0, 10, 9.1]])
# target = torch.tensor([[-1.0, 0, 10, 1.2], [-1.0, 0, 10, 2.2], [1.0, 0, 10, 1.1]])
# print("loss: ", criterion(output, target))


print("=== FocalLoss ===")
# 使用softmax，适用于多分类
criterion = FocalLoss()

output = torch.tensor([[-1.0, 0, 120, 1.2], [1.0, -10, 1.0, 1.2], [9.0, 0, -10, 9.1]])
target = torch.tensor([-100, 1, 2])
print("loss: ", criterion(output, target), criterion(output, target).requires_grad)


# print("=== CosineEmbeddingLoss ===")
# # pairwise, 基于余弦相似度来衡量两个样本之间的相似性
# # target=1: 1 - cos(x1, x2)
# # target=-1: max(cos(x1, x2)-margin, 0)
# # 优化目标：当target=1时，让x1和x2相似，当target=-1时，让x1和x2不相似
# # 预测时：二分类：x1和x2相似给label正；x1和x2不相似给label负
# criterion = nn.CosineEmbeddingLoss(margin=0)

# x1 = torch.tensor([[0.0, 1, 2]])
# x2 = torch.tensor([[0.0, 1, 2]])
# target = torch.tensor([1])
# print("loss应该小: ", criterion(x1, x2, target), "cos 相似性: ", nn.CosineSimilarity(dim=-1)(x1, x2))
# target = torch.tensor([-1])
# print("loss应该大: ", criterion(x1, x2, target), "cos 相似性: ", nn.CosineSimilarity(dim=-1)(x1, x2))

# x1 = torch.tensor([[10.0, 10, 0]])
# x2 = torch.tensor([[-10.0, -10, 0]])
# target = torch.tensor([-1])
# print("loss应该小: ", criterion(x1, x2, target), "cos 相似性: ", nn.CosineSimilarity(dim=-1)(x1, x2))
# target = torch.tensor([1])
# print("loss应该大: ", criterion(x1, x2, target), "cos 相似性: ", nn.CosineSimilarity(dim=-1)(x1, x2))


# print("=== MarginRankingLoss ===")
# # pairwise，2个样本排序的loss
# # loss = max(-y(x1-x2)+margin, 0)
# # 优化目标：target=1时，让x1>x2; 当target=-1时，让x1<x2
# # 预测时：二分类：x1>x2时给label正, x1<x2时给label负

# criterion = nn.MarginRankingLoss(margin=0.0)

# target = torch.tensor([-1, -1, 1])
# x1 = torch.tensor([1.0, 3, 5])
# x2 = torch.tensor([2.0, 4, 3])
# print("loss应该小: ", criterion(x1, x2, target), "x1-x2: ", x1 - x2)
# x1 = torch.tensor([1.0, 3, 5])
# x2 = torch.tensor([0.0, 2, 7])
# print("loss应该大: ", criterion(x1, x2, target), "x1-x2: ", x1 - x2)


# print("=== KLDivLoss ===")
# # KL散度
# # loss = y_true * log(y_true / y_pred)
# # 优化目标：让y_true 和 y_pred相同/信息损失最小
# criterion = nn.KLDivLoss(reduction="batchmean")

# target = torch.tensor([[0.0, 1, 2]])
# output = torch.tensor([[0.0, 1, 2]])
# print("loss应该小: ", criterion(output, target))
# output = torch.tensor([[-10.0, 10, -10]])
# print("loss应该大: ", criterion(output, target))


# print("=== TripletMarginWithDistanceLoss ===")
# # Triplet
# # 优化目标：让正样本和anchor接近, 负样本和anchor远离, 并可自定义距离函数
# # loss = max(d(a,p) - d(a,n) + margin, 0)
# # 预测时：二分类：d(a,p) < d(a,n)时给label正，d(a,p) > d(a,n)时给label负
# # margin：控制正负样本之间的最小距离，margin过大会导致模型过于保守，有效的三元组也难以被正确学习；margin过小可能导致模型过于激进，无效的三元组也被错误地学习

# criterion = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(p=2))  # 与锚点的p范数距离
# anchor = torch.tensor([[-3.0, -1, 0, 1, 3]])
# positive = torch.tensor([[-3.0, -1, 0, 1, 3]])
# negative = torch.tensor([[3.0, 1, 0, -1, -3]])
# print("loss应该小: ", criterion(anchor, positive, negative))
# anchor = torch.tensor([[-3.0, -1, 0, 1, 3]])
# positive = torch.tensor([[3.0, 1.3, 0, -1, -3]])
# negative = torch.tensor([[-3.0, -1, 0, 1, 3]])
# print("loss应该大: ", criterion(anchor, positive, negative))

# distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)
# criterion = nn.TripletMarginWithDistanceLoss(distance_function=distance_function, margin=1.0)  # 余弦相似距离[-1, 1]

# anchor = torch.tensor([[-3.0, -1, 0, 1, 3]])
# positive = torch.tensor([[-3.0, -1, 0, 1, 3]])
# negative = torch.tensor([[3.0, 1, 0, -1, -3]])
# print("loss应该小: ", criterion(anchor, positive, negative), distance_function(anchor, positive) < distance_function(anchor, negative))
# anchor = torch.tensor([[-3.0, -1, 0, 1, 3]])
# positive = torch.tensor([[3.0, 1.3, 0, -1, -3]])
# negative = torch.tensor([[-3.0, -1, 0, 1, 3]])
# print("loss应该大: ", criterion(anchor, positive, negative), distance_function(anchor, positive) > distance_function(anchor, negative))


# print("=== InfoNCE ===")
# # 2组样本排成矩阵, 对角线相似, 非对角线不相似


# class InfoNCELoss(nn.Module):
#     def __init__(self, temperature=0.5) -> None:
#         super().__init__()
#         self.temperature = temperature
        
#     def forward(self, proj_1, proj_2):
#         # Calculate cosine similarity
#         cos_sim = nn.CosineSimilarity(dim=-1)(proj_1.unsqueeze(1), proj_2.unsqueeze(0))
#         # InfoNCE loss
#         loss = -nn.LogSoftmax(dim=-1)(cos_sim / self.temperature).diag().mean()
#         return loss, cos_sim


# criterion = InfoNCELoss()
# feats1 = torch.tensor([[1.0, 2, 3, 4, 5], [-1, 12, 43, 134, -5], [2321, 221, 33, -4, -25]])
# feats2 = torch.tensor([[1.0, 2, 3, 4, 5], [-1, 12, 43, 134, -5], [2321, 221, 33, -4, -25]])
# print("loss应该小: ", criterion(feats1, feats2))
# feats1 = torch.tensor([[1.0, 2, 3, 4, 5], [-1, 12, 43, 134, -5], [2321, 221, 33, -4, -25]])
# feats2 = torch.tensor([[-1.0, -2, -3, -4, -5], [1, -12, -43, -134, 5], [-2321, -221, -33, 4, 25]])
# print("loss应该大: ", criterion(feats1, feats2))
