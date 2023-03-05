import torch
from transformers import BertTokenizer, BertModel, BertConfig


pretrained_model_name = "hfl/chinese-roberta-wwm-ext-large"
config = BertConfig.from_pretrained(pretrained_model_name)
config.output_hidden_states = True
config.return_dict = False

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

model = BertModel.from_pretrained(pretrained_model_name, config=config)

text1 = "北京，上海"
text2 = "北京，天津, 上海"
maxlen = 7

tokens = tokenizer.tokenize(text1)  # 对文本分词
print("tokens: ", tokens)
print("id: ", tokenizer.convert_tokens_to_ids(tokens))  # token转id
print("decode: ", tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens)))  # 输入input_ids, 可以进行反向解码

input1 = tokenizer(
    text1, text2,  # [cls]句子1[sep]句子2[sep]
    padding='max_length',  # 不足时填充到指定最大长度
    truncation=True,  # 过长时截断
    max_length=maxlen,  # 2个句子加起来的长度
    return_tensors='pt')  # 返回字典, input_ids, token_type_ids, attention_mask
print("input1: ", input1)

sequence_output, pooled_output, hidden_states = model(**input1)
print(sequence_output.shape)
print(pooled_output.shape)
print(len(hidden_states), hidden_states[0].shape, hidden_states[1].shape)


text3 = "长白山"

input3 = tokenizer(
    text3,  # [cls]句子1[sep]句子2[sep]
    padding='max_length',  # 不足时填充到指定最大长度
    truncation=True,  # 过长时截断
    max_length=maxlen,  # 2个句子加起来的长度
    return_tensors='pt')  # 返回字典, input_ids, token_type_ids, attention_mask
print("input3: ", input3)

sequence_output2, pooled_output2, hidden_states2 = model(**input3)

print("维度: ", pooled_output2.shape)
print("相似度: ", torch.cosine_similarity(pooled_output, pooled_output2, dim=1, eps=1e-08))
