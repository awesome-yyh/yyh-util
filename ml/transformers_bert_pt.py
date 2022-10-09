import torch
from transformers import BertTokenizer, BertModel, BertConfig


pretrained_model_name = "hfl/chinese-roberta-wwm-ext-large"
config = BertConfig.from_pretrained(pretrained_model_name, output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, config=config)
model = BertModel.from_pretrained(pretrained_model_name, config=config)

text1 = "北京"
input1 = tokenizer(text1, return_tensors='pt')
output1 = model(**input1)[-2]

text2 = "长白山"
input2 = tokenizer(text2, return_tensors='pt')
output2 = model(**input2)[-2]

print("维度: ", output2.shape)
print("相似度: ", torch.cosine_similarity(output1, output2, dim=1, eps=1e-08))
