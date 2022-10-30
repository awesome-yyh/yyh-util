import tokenization


# 对文本编码
str = "大哥大"
max_seq_length = 12
tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt")

tokens = tokenizer.tokenize(str)
print(tokens)

# Account for [CLS] and [SEP] with "- 2"
if len(tokens) > max_seq_length - 2:
    tokens = tokens[:max_seq_length - 2]

strtoken = ["[CLS]"]
strtoken.extend(tokens)
strtoken.append("[SEP]")
print(strtoken, type(strtoken))

input_ids = tokenizer.convert_tokens_to_ids(strtoken)

while len(input_ids) < max_seq_length:
    input_ids.append(0)

print(input_ids, type(input_ids), len(input_ids))
