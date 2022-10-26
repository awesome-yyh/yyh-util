import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util


def get_cos_similar_multi(v1: list, v2: list):
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

pretrained_model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
model = SentenceTransformer(pretrained_model_name) # 会自动下载模型
t1 = time.time()
query_embedding = model.encode('北京')
t2 = time.time()
print(t2-t1)
docs_embedding = model.encode(['长白山','故宫', '北京故宫', '北京'])
print(time.time()-t2)

print(query_embedding.shape)
print(docs_embedding.shape)
print("相似度: ", get_cos_similar_multi(query_embedding.tolist(), docs_embedding.tolist()))
print("Similarity:", util.pytorch_cos_sim(query_embedding, docs_embedding))
