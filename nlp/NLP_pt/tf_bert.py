from transformers import BertTokenizer, TFBertModel, BertConfig
import tensorflow as tf


def tf_cosine_distance(tensor1, tensor2):
	"""
	consine相似度：用两个向量的夹角判断两个向量的相似度，夹角越小，相似度越高，得到的consine相似度数值越大
	数值范围[-1,1],数值越大越相似。
	:param tensor1:
	:param tensor2:
	:return:
	"""
    # 把张量拉成矢量，这是我自己的应用需求
	tensor1 = tf.reshape(tensor1, shape=(1, -1))
	tensor2 = tf.reshape(tensor2, shape=(1, -1))
	
	# 求模长
	tensor1_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor1)))
	tensor2_norm = tf.sqrt(tf.reduce_sum(tf.square(tensor2)))
	
	# 内积
	tensor1_tensor2 = tf.reduce_sum(tf.multiply(tensor1, tensor2))
	cosin = tensor1_tensor2 / (tensor1_norm * tensor2_norm)
	
	return cosin


pretrained_model_name = "hfl/chinese-roberta-wwm-ext"
config = BertConfig.from_pretrained(pretrained_model_name, output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, config=config)
model = TFBertModel.from_pretrained(pretrained_model_name, config=config)

text1 = "北京"  # 当文本速度过长时，时间会过长
input1 = tokenizer(text1, return_tensors='tf')
print(input1)
output1 = model(input1)[-2]

text2 = "长白山北部"
input2 = tokenizer(text2, return_tensors='tf')
print(input2)  # input_ids, token_type_ids, attention_mask
output2 = model(input2)[-2]

print("维度: ", output2.shape)
print("相似度: ", tf_cosine_distance(output1, output2))
