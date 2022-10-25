from PIL import Image
import requests
# import clip
import torch
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import CLIPProcessor, CLIPModel
import numpy as np


def get_cos_similar_multi(v1: list, v2: list):
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res

txt_pretrained_model_name = "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese"
text_tokenizer = BertTokenizer.from_pretrained(txt_pretrained_model_name)
text_encoder = BertForSequenceClassification.from_pretrained(txt_pretrained_model_name).eval()

query_texts = ['北京', '长白山','故宫', '北京故宫', '北京']  # 这里是输入文本的，可以随意替换。
text = text_tokenizer(query_texts, return_tensors='pt', padding=True)['input_ids']

with torch.no_grad():
    text_features = text_encoder(text).logits
    print(text_features.shape)
    # 归一化
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # print(text_features[0].tolist())
    # 计算余弦相似度
    print("相似度: ", get_cos_similar_multi(text_features[0].tolist(), text_features.tolist()))
    # print(query_texts[np.argmax(probs)])
