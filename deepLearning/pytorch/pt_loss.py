import torch
import torch.nn as nn
import torch.nn.functional as F


torch.random.manual_seed(3)

print("=== MSELoss ===")
# 输入和Label的均方差
# loss = mean((x-y)^2)
criterion = nn.MSELoss()

output = torch.randn(3, 5)
target = torch.randn(3, 5)

print(criterion(output, target))


print("=== BCEWithLogitsLoss ===")
# 使用sigmoid，适用于2分类、多标签分类
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0, 1.5, 2, 1.0]))  # pos_weight: 每个类别的正例的权重, 通常设为设为负样本数/正样本数

output = torch.full([10, 4], 1.5)
target = torch.ones([10, 4])  # 4个类别

print(criterion(output, target))


print("=== CrossEntropyLoss ===")
# 使用softmax，适用于多分类
criterion = nn.CrossEntropyLoss()

output = torch.randn(3, 5)
target = torch.ones([3, 5])

print(criterion(output, target))


print("=== KLDivLoss ===")
# KL散度
criterion = nn.KLDivLoss(reduction="batchmean")

output = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
target = F.softmax(torch.rand(3, 5), dim=1)

print(criterion(output, target))


print("=== CosineEmbeddingLoss ===")
# pairwise, 基于余弦相似度衡量两个样本之间的相似性
# 目标是使相似的样本之间的余弦相似度接近1，不相似的样本之间的余弦相似度接近-1
# 样本是文本向量、图像向量等
# 输入相似样本对，并给label=1; 或不相似样本对，并给label=-1
# label=1: 1 - cos(x1, x2)
# label=-1: max(cos(x1, x2)-margin, 0)
criterion = nn.CosineEmbeddingLoss()
x1 = torch.tensor([[1.0617, 1.3397, -0.2303],
                  [0.3459, -0.9821, 1.2511]])
x2 = torch.tensor([[1.0617, 1.3397, -0.2303],
                  [0.4486, -0.6504, 1.5173]])
target = torch.tensor([1, -1])

print(criterion(x1, x2, target))


print("=== MarginRankingLoss ===")
# pairwise, 比较两个样本之间的距离来衡量它们之间的相似性
# 目标是使相似的样本之间的距离尽可能小，而不相似的样本之间的距离尽可能大
# 样本是图像之间的欧氏距离、文本之间的编辑距离等
# y=1时，第一个样本排在前；y=-1时，第二个样本排在前
# loss = max(-y(x1-x2)+margin, 0)
criterion = nn.MarginRankingLoss(margin=0.2)
x1 = torch.tensor([0.8, 0.4, 0.1])
x2 = torch.tensor([0.6, 0.3, 0.2])
target = torch.tensor([1, 1, -1])

print(criterion(x1, x2, target))


print("=== TripletMarginWithDistanceLoss ===")
# Triplet, 正样本接近anchor，负样本远离anchor, 并可自定义距离函数
# max(d(a,p) - d(a,n) + margin, 0)


def cosine_distance(x1, x2):
    """
    相似趋于0, 不相似趋于1
    """
    cosine_similarity = nn.CosineSimilarity(dim=1)  # 余弦相似距离
    print("cosine_similarity: ", cosine_similarity(x1, x2))
    return (1 - cosine_similarity(x1, x2)) / 2


criterion = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=1.0)
# distance_function=nn.PairwiseDistance(p=2)  # 与锚点的p范数距离
# cosine_distance # 与锚向量的余弦距离

positive = torch.tensor([[1.0617, 1.3397, -0.2303],
                        [0.3459, -0.9821, 1.2511]])
negative = torch.tensor([[-2.0617, -0.3397, 0.82303],
                        [-0.3459, 0.9821, -1.2511]])
anchor = positive
print("cosine_distance: ", cosine_distance(positive, positive))
print("triplet loss: ", criterion(positive, positive, negative))
print("not cosine_distance: ", cosine_distance(positive, negative))
print("not triplet loss: ", criterion(negative, positive, negative))
# anchor = torch.ones_like(positive)  # 锚点
anchor = torch.tensor([[-1.0617, -1.3397, -1.2303],
                      [1.3459, 1.9821, -0.2511]])
print(criterion(anchor, positive, negative))

nn.TripletMarginWithDistanceLoss
print("=== InfoNCE ===")


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5) -> None:
        super().__init__()
        self.temperature = temperature
        
    def forward(self, proj_1, proj_2):
        # Calculate cosine similarity
        cos_sim = nn.CosineSimilarity(dim=-1)(proj_1.unsqueeze(1), proj_2.unsqueeze(0))
        # InfoNCE loss
        loss = -nn.LogSoftmax(dim=1)(cos_sim / self.temperature).diag().mean()
        return cos_sim, loss


criterion = InfoNCELoss()
feats1 = torch.tensor([[1, 2, 3, 4, 5], [6, 6, 7, 8, 9], [10, 13, 10, 11, 12]], dtype=torch.float32)
feats2 = torch.tensor([[1, 2, 3, 4, 5], [6, 5, 7, 8, 9], [10, 13, 10, 11, 13]], dtype=torch.float32)

print(criterion(feats1, feats2))

feats1 = torch.tensor([[1, 2, 3, 4, 5], [-1, 12, 43, 134, -5], [2321, 221, 33, -4, -25]], dtype=torch.float32)
feats2 = torch.tensor([[1, 2, 3, 4, 5], [-1, 12, 43, 134, -5], [2321, 221, 33, -4, -25]], dtype=torch.float32)

print(criterion(feats1, feats2))
