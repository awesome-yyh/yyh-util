import torch
from transformers import BertTokenizer, BertModel, BertConfig


pretrained_model_name = "hfl/chinese-roberta-wwm-ext"

config = BertConfig.from_pretrained(pretrained_model_name)
print("config: ", config, type(config))

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

model = BertModel.from_pretrained(pretrained_model_name, config=config)

text1 = "北京，上海"
text2 = "南京，上海，天津"
maxlen = 11

print("*** tokenizer基本使用 ***")
print("tokenizer len: ", len(tokenizer))

tokens = tokenizer.tokenize(text1)  # 句子转tokens
print("tokens: ", tokens)

string = tokenizer.convert_tokens_to_string(tokens)  # tokens转句子（空格连接各个token）
print("string: ", string)

id = tokenizer.convert_tokens_to_ids(tokens)  # token转id
print("id: ", id)

id = tokenizer.encode(text1)  # 直接句子转id, 会添加上[cls]等标识，相当于仅返回input_ids
print("input_ids: ", id)

print("decode: ", tokenizer.decode(id))  # id解码得到token

# 添加新的token
# 1. 将vocab.txt中的[unusedxx]改写成需要添加的token
# 2. tokenizer.add_special_tokens 如下：
tokenizer.add_special_tokens({
    "additional_special_tokens": ["[ENTITY]"]})
print("tokenizer len: ", len(tokenizer))
# add_special_tokens不会被分割
# 如果vocab.txt中有这个token, 则tokenizer长度不改，不需要model.resize_token_embeddings(len(tokenizer))
# 如果vocab.txt中没有这个token，tokenizer长度会增加，需要model.resize_token_embeddings(len(tokenizer))
# 最后tokenizer.save_pretrained(<output_dir>)，保存以备再次使用

print("*** bert形式的输入 ***")
# tokenizer.encode_plus() 和 tokenizer.batch_encode_plus
# 已经被弃用，都统一使用__call__方法
# 一个句子 text1
# 一对句子 text1, text2
# 多个句子 [text1, text2]
# 多对句子 [[text1, text2], [text1, text2]]
bert_input = tokenizer(
    [[text1, text2]],  # [cls]句子1[sep]句子2[sep]
    padding='max_length',  # 不足时填充到指定最大长度
    truncation=True,  # 过长时截断
    max_length=maxlen,  # 2个句子加起来的长度
    return_tensors='pt')  # 返回字典, input_ids, token_type_ids, attention_mask
print("bert_input: ", bert_input)

# 获取某个token的位置索引
cls_token_index = torch.where(bert_input["input_ids"] == tokenizer.cls_token_id)

# entity_outputs = outputs[0][cls_token_indices].reshape(outputs[0].shape[0], -1, outputs[0].shape[2])  # 获取所有[CLS] token的输出

print(tokenizer.cls_token_id, cls_token_index)

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
