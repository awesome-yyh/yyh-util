import numpy as np
from transformers import BertTokenizer, BertModel
import torch


def get_cos_similar_multi(v1: list, v2: list):
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"正在使用的设备：{device}")

# Load model from HuggingFace Hub
tokenizer = BertTokenizer.from_pretrained('/data/app/base_model/shibing624-text2vec-base-chinese')
model = BertModel.from_pretrained('/data/app/base_model/shibing624-text2vec-base-chinese')
model.to(device)
model.eval()

sentences = ['如何更换花呗绑定银行卡', '红红火火']
# Tokenize sentences
encoded_input = tokenizer(sentences, max_length=128, padding=True, truncation=True, return_tensors='pt').to(device)

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
print("Sentence embeddings:")
print(sentence_embeddings, sentence_embeddings.shape)

print("相似度: ", get_cos_similar_multi([sentence_embeddings[0].tolist()], [sentence_embeddings[1].tolist()]))
