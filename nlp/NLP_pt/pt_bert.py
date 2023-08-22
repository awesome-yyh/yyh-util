import torch
from transformers import BertTokenizer, BertModel, BertConfig


pretrained_model_name = "hfl/chinese-roberta-wwm-ext"

config = BertConfig.from_pretrained(pretrained_model_name)
print("config: ", config, type(config))

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

model = BertModel.from_pretrained(pretrained_model_name, config=config)

print("*** bert model 结构 ***")
params = list(model.named_parameters())
print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


sequence_output, pooled_output, hidden_states = model(**bert_input, output_hidden_states=True, return_dict=False)  # model的输入必须以batch形式输入即input_ids等都是二维向量

print("对token操作使用sequence_output: ", sequence_output.shape)
print("对整个句子操作使用pooled_output: ", pooled_output.shape)
print("hidden: ", len(hidden_states), hidden_states[0].shape)


text3 = "长白山"

input3 = tokenizer(
    text3,  # [cls]句子1[sep]句子2[sep]
    padding='max_length',  # 不足时填充到指定最大长度
    truncation=True,  # 过长时截断
    max_length=maxlen,  # 2个句子加起来的长度
    return_tensors='pt')  # 返回字典, input_ids, token_type_ids, attention_mask
print("input3: ", input3)

sequence_output2, pooled_output2, hidden_states2 = model(**input3, output_hidden_states=True, return_dict=False)

print("维度: ", pooled_output2.shape)
print("相似度: ", torch.cosine_similarity(pooled_output, pooled_output2, dim=1, eps=1e-08))
