import torch
import torch.nn as nn
import torch.nn.functional as F


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


print("=== KLDivLoss ===")
# KL散度
criterion = nn.KLDivLoss(reduction="batchmean")

target = torch.tensor([[0.0, 1, 2]])
output = torch.tensor([[0.0, 1, 2]])
print("loss应该小: ", criterion(output, target))
output = torch.tensor([[-10.0, 10, -10]])
print("loss应该大: ", criterion(output, target))


print("=== CosineEmbeddingLoss ===")
# pairwise, 基于余弦相似度衡量两个样本之间的相似性
# 2个样本相似给label1, 不相似给label-1
# label=1: 1 - cos(x1, x2)
# label=-1: max(cos(x1, x2)-margin, 0)
criterion = nn.CosineEmbeddingLoss(margin=0.0)

target = torch.tensor([1, -1])
x1 = torch.tensor([[0.0, 1, 2], [-10.0, 10, 0]])
x2 = torch.tensor([[0.0, 1, 2], [10, -10, 0]])
print("loss应该小: ", criterion(x1, x2, target))
x1 = torch.tensor([[0.0, 1, 2], [-10.0, 10, 0]])
x2 = torch.tensor([[0.0, -1, -2], [-10, 10, 0]])
print("loss应该大: ", criterion(x1, x2, target))


print("=== MarginRankingLoss ===")
# pairwise，2个样本排序的loss
# loss = max(-y(x1-x2)+margin, 0)
# y=1时，x1大时loss=0，x1小时loss大，优化目标是使x1 > x2, y=-1时则相反
# x1>x2时给label1, x1<x2时给label-1
criterion = nn.MarginRankingLoss(margin=0.0)

target = torch.tensor([-1, -1, 1])
x1 = torch.tensor([1, 3, 5])
x2 = torch.tensor([2, 4, 3])
print("loss应该小: ", criterion(x1, x2, target))
x1 = torch.tensor([1, 3, 5])
x2 = torch.tensor([0, 2, 7])
print("loss应该大: ", criterion(x1, x2, target))


print("=== TripletMarginWithDistanceLoss ===")
# Triplet, 正样本和anchor接近, 负样本和anchor远离, 并可自定义距离函数
# max(d(a,p) - d(a,n) + margin, 0)


def cosine_distance(x1, x2):
    """相似趋于0, 不相似趋于1"""
    cosine_similarity = nn.CosineSimilarity(dim=-1)  # 余弦相似距离
    print("cosine_similarity: ", cosine_similarity(x1, x2))
    return (1 - cosine_similarity(x1, x2)) / 2


criterion = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=1.0)
# distance_function=nn.PairwiseDistance(p=2)  # 与锚点的p范数距离
# distance_function=cosine_distance # 与锚向量的余弦距离

anchor = torch.tensor([[-3.0, -1, 0, 1, 3]])
positive = torch.tensor([[-3.0, -1, 0, 1, 3]])
negative = torch.tensor([[3.0, 1, 0, -1, -3]])
print("loss应该小: ", criterion(anchor, positive, negative))
anchor = torch.tensor([[-3.0, -1, 0, 1, 3]])
positive = torch.tensor([[3.0, 1, 0, -1, -3]])
negative = torch.tensor([[-3.0, -1, 0, 1, 3]])
print("loss应该大: ", criterion(anchor, positive, negative))

print("=== InfoNCE ===")
# 2组样本排成矩阵, 对角线相似, 非对角线不相似


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5) -> None:
        super().__init__()
        self.temperature = temperature
        
    def forward(self, proj_1, proj_2):
        # Calculate cosine similarity
        cos_sim = nn.CosineSimilarity(dim=-1)(proj_1.unsqueeze(1), proj_2.unsqueeze(0))
        # InfoNCE loss
        loss = -nn.LogSoftmax(dim=-1)(cos_sim / self.temperature).diag().mean()
        return cos_sim, loss


criterion = InfoNCELoss()
feats1 = torch.tensor([[1.0, 2, 3, 4, 5], [-1, 12, 43, 134, -5], [2321, 221, 33, -4, -25]])
feats2 = torch.tensor([[1.0, 2, 3, 4, 5], [-1, 12, 43, 134, -5], [2321, 221, 33, -4, -25]])
print("loss应该小: ", criterion(feats1, feats2))
feats1 = torch.tensor([[1.0, 2, 3, 4, 5], [-1, 12, 43, 134, -5], [2321, 221, 33, -4, -25]])
feats2 = torch.tensor([[-1.0, -2, -3, -4, -5], [1, -12, -43, -134, 5], [-2321, -221, -33, 4, 25]])
print("loss应该大: ", criterion(feats1, feats2))
