import numpy as np
import tensorflow as tf
import tokenization


# one-hot编码
output = tf.one_hot(3, depth=10)
print("将3转成one-hot形式: ", output)

# embedding
emb1 = tf.keras.layers.Embedding(input_dim=len(input_ids),output_dim=10)(np.array(input_ids))


# 对文本编码
tokenizer = tokenization.FullTokenizer(vocab_file="vocab.txt")
print(tokenizer)

str = "大哥大"
token = tokenizer.tokenize(str)
strtoken = ["[CLS]"]
strtoken.extend(token)
strtoken.append("[SEP]")
print(strtoken, type(strtoken))

input_ids = tokenizer.convert_tokens_to_ids(strtoken)
print(input_ids, type(input_ids))

emb1 = tf.keras.layers.Embedding(input_dim=len(input_ids),output_dim=10)(np.array(input_ids))
print('result 1:',emb1)


# feature = InputFeatures(
#       input_ids=input_ids,
#       input_mask=input_mask,
#       segment_ids=segment_ids,
#       label_id=label_id,
#       is_real_example=True)
# print(emb2)