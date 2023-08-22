import torch
import torch.nn as nn
import torch.nn.functional as F


print("=== MSELoss ===")
# 2个元素的均方差
loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
print(output)


print("=== BCEWithLogitsLoss ===")
# 使用sigmoid，适用于2分类、多标签分类
target = torch.ones([10, 4], dtype=torch.float32)  # 64 classes, batch size = 10
output = torch.full([10, 4], 1.5)  # A prediction (logit)
pos_weight = torch.ones([4])  # All weights are equal to 1
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 正例的权重，长度等于类别数，大于1增加召回率，小于1增加准确率，通常在正负样本不均衡时使用，设为负样本数/正样本数，例如100正样本300负样本，pos_weight=300/100=3
print(output, target, criterion(output, target))

print("=== CrossEntropyLoss ===")
# 使用softmax，适用于多分类
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
print(input, target, output)
# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
print(input, target, output)


print("=== KLDivLoss ===")
# KL散度
kl_loss = nn.KLDivLoss(reduction="batchmean")
# input should be a distribution in the log space
input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
# Sample a batch of distributions. Usually this would come from the dataset
target = F.softmax(torch.rand(3, 5), dim=1)
output = kl_loss(input, target)
print(output)
kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
log_target = F.log_softmax(torch.rand(3, 5), dim=1)
output = kl_loss(input, log_target)
print(output)


print("=== CosineSimilarity ===")
# 返回2个元素的余弦相似距离
input1 = torch.randn(100, 128)
input2 = torch.randn(100, 128)
cos = nn.CosineSimilarity(dim=1, eps=1e-6)  # x*y / max(x的2范数 * y的2范数，e)
output = cos(input1, input2)
print(output)

print("=== PairwiseDistance ===")
pdist = nn.PairwiseDistance(p=2)  # ||x-y+e||的p范数
input1 = torch.tensor([[[1, 2, 3], [3, 4, 5]]], dtype=torch.float32)
input2 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
output = pdist(input1, input2)
print(input1, input2, output)
