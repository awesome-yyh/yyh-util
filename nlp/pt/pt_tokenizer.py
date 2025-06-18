import torch
from transformers import BertTokenizer, BertConfig


pretrained_model_name = "hfl/chinese-roberta-wwm-ext"

config = BertConfig.from_pretrained(pretrained_model_name)
print("config: ", config, type(config))

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

text1 = "北京，上#海[MASK]"
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
