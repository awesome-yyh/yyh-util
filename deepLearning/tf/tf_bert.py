import json
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd


# 对bert fune-tuning后做文本分类

# 数据读取
df_raw = pd.read_csv("deepLearning/tf/data.txt",sep="\t",header=None,names=["text","label"])

# 数据探索分析

# 数据清洗(缺失值、重复值、异常值、大小写、标点)

# 数据采样(搜集、合成、过采样、欠采样、阈值移动、loss加权、评价指标)

# 特征工程(数值、文本、类别、时间)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def bert_input_encode(sentence, max_length = 20):
    # add special tokens
    test_sentence_with_special_tokens = '[CLS]' + sentence + '[SEP]'
    tokenized = tokenizer.tokenize(test_sentence_with_special_tokens)
    # print('tokenized', tokenized)

    # convert tokens to ids in WordPiece
    input_ids = tokenizer.convert_tokens_to_ids(tokenized)

    # precalculation of pad length, so that we can reuse it later on
    padding_length = max_length - len(input_ids)

    # map tokens to WordPiece dictionary and add pad token for those text shorter than our max length
    input_ids = input_ids + ([0] * padding_length)

    # attention should focus just on sequence with non padded tokens
    attention_mask = [1] * len(input_ids)

    # do not focus attention on padded tokens
    attention_mask = attention_mask + ([0] * padding_length)

    # token types, needed for example for question answering, for our purpose we will just set 0 as we have just one sequence
    token_type_ids = [0] * max_length
    bert_input = {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask
    } 
    return bert_input


# map to the expected input to TFBertForSequenceClassification, see here 
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)

    for index, row in ds.iterrows():
        review = row["text"]
        label = row["y"]
        bert_input = bert_input_encode(review)

        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

df_label = pd.DataFrame({"label":["财经","房产","股票","教育","科技","社会","时政","体育","游戏","娱乐"],"y":list(range(10))})
df_raw = pd.merge(df_raw,df_label,on="label",how="left")

# 划分训练集、验证集、测试集
def split_dataset(df):
    train_set, x = train_test_split(df, 
        stratify=df['label'],
        test_size=0.1, 
        random_state=42)
    val_set, test_set = train_test_split(x, 
        stratify=x['label'],
        test_size=0.5, 
        random_state=43)

    return train_set,val_set, test_set
train_data, val_data, test_data = split_dataset(df_raw)

batch_size = 128
# ds_train_encoded = encode_examples(train_data).shuffle(10000).batch(batch_size)
# ds_val_encoded = encode_examples(val_data).batch(batch_size)
# ds_test_encoded = encode_examples(test_data).batch(batch_size)

# 存储为TFRecord

# 将tfrecords数据解析出来

#读取tfrecords文件

# 搭建模型
model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=10)

# with open('bert_config.json','r') as f:
#     config = json.load(f)

# seq_len = config['max_position_embeddings']
# unique_id  = tf.keras.Input(shape=(1,),dtype=tf.int64,name='unique_id')
# input_ids   = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_ids')
# input_mask  = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='input_mask')
# segment_ids = tf.keras.Input(shape=(seq_len,),dtype=tf.int32,name='segment_ids')
# BERT = modeling.BertModel(config=config,name='bert')
# pooled_output, sequence_output = BERT(input_word_ids=input_ids,
#                                         input_mask=input_mask,
#                                         input_type_ids=segment_ids)
# 1/0
# logits = TDense(2,name='logits')(sequence_output)
# start_logits,end_logits = tf.split(logits,axis=-1,num_or_size_splits= 2,name='split')
# start_logits = tf.squeeze(start_logits,axis=-1,name='start_squeeze')
# end_logits   = tf.squeeze(end_logits,  axis=-1,name='end_squeeze')

# ans_type      = TDense(5,name='ans_type')(pooled_output)
# model = tf.keras.Model([input_ for input_ in [unique_id,input_ids,input_mask,segment_ids] 
#                         if input_ is not None],
#                         [unique_id,start_logits,end_logits,ans_type],
#                         name='bert-baseline')   

# 查看模型结构
# model.build((None,))
model.summary()
tf.keras.utils.plot_model(model, "bert.png", show_shapes=True)

# 定义损失函数及优化器
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss = tf.keras.losses.sparse_categorical_crossentropy, 
              metrics=['accuracy'])


# 训练模型
# early_stopping