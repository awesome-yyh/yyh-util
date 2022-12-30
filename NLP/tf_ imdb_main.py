import os, datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from prepare_data import convert_raw_data_into_csv, data_preprocess, data_preprocess_bert
from tf_fasttext import FastText
from tf_textcnn import TextCNN
from tf_textrnn import TextRNN
from tf_textSelfAtt import TextSelfAtt
from tf_transformer import TextTransformerEncoder
from tf_bert import bert_model


"""
imdb电影评价二分类
"""

# 读取数据
data_df = convert_raw_data_into_csv()
print(data_df.shape)
print(data_df.head(10))

# 数据探索分析

# 数据清洗(缺失值、重复值、异常值、大小写、标点)

# 类别不均衡(搜集、合成、过采样、欠采样、阈值移动、loss加权、更改评价指标)
print(data_df['label'].value_counts(normalize=True))

# 特征工程(数值、类别、时间、文本、图像)
train_dataset, val_dataset, test_dataset = data_preprocess(data_df)

# 特征工程(数值、类别、时间、文本、图像)(bert)
# pretrained_model_name, from_pt = "hfl/chinese-roberta-wwm-ext", False

# pretrained_model_name, from_pt = "prajjwal1/bert-tiny", True # (L=2, H=128)
# pretrained_model_name, from_pt = "prajjwal1/bert-mini", True # (L=4, H=256)
# pretrained_model_name, from_pt = "prajjwal1/bert-small", True # (L=4, H=512)
# pretrained_model_name, from_pt = "prajjwal1/bert-medium", True # (L=8, H=512) 

# pretrained_model_name, from_pt = "google/bert_uncased_L-2_H-128_A-2", True # (BERT-Tiny) uncased代表需要转小写字母再用
# pretrained_model_name, from_pt = "google/bert_uncased_L-2_H-768_A-12", True # (BERT-Mini)
# pretrained_model_name, from_pt = "google/bert_uncased_L-4_H-512_A-8", True # (BERT-Small)
# pretrained_model_name, from_pt = "google/bert_uncased_L-8_H-512_A-8", True # (BERT-Medium)
# pretrained_model_name, from_pt = "google/bert_uncased_L-12_H-768_A-12", True # (BERT-Base)

# train_dataset, val_dataset, test_dataset = data_preprocess_bert(data_df, pretrained_model_name, from_pt)

# 搭建模型
seq_len = 100
max_words=10000
model = FastText(maxlen=seq_len,
                max_features=max_words,
                embedding_dims=100,
                class_num=2,
                last_activation='softmax')

# model = TextCNN(maxlen=seq_len,
#                 max_features=max_words,
#                 embedding_dims=200,
#                 class_num=2,
#                 kernel_sizes=[2,3,5],
#                 kernel_regularizer=None,
#                 last_activation='softmax')

# model = TextRNN(maxlen=seq_len,
#                 max_features=max_words,
#                 embedding_dims=100,
#                 class_num=2,
#                 last_activation='softmax'
#                 )

# model = TextSelfAtt(maxlen=seq_len,
#                     max_features=max_words,
#                     embedding_dims=400,
#                     num_heads=4,
#                     class_num=2,
#                     last_activation='softmax'
#                     )

# model = TextTransformerEncoder(maxlen=seq_len,
#                     max_features=max_words,
#                     embedding_dims=400,
#                     num_heads=2,
#                     class_num=2,
#                     last_activation='softmax'
#                     )

# model = bert_model(pretrained_model_name=pretrained_model_name,
#                     seq_len=seq_len, 
#                     class_num=2, 
#                     from_pt=from_pt,
#                     last_activation='softmax'
#                     )

# # 查看模型结构
# model.build(input_shape=(None, seq_len))
# model.summary()
# tf.keras.utils.plot_model(model.build_graph(seq_len), "NLP/fasttext.png",
#                         show_shapes=True)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss = {"output_1": "categorical_crossentropy"},
              loss_weights=[1.0], # 为每个损失分配不同的权重
              metrics=['accuracy'])

# 训练模型
# early_stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    verbose=1,
    patience=10, # 每10步检查一下是否提升
    mode='max', # monitor是准确率时max，是损失时min
    restore_best_weights=True)

# tensorboard
log_dir="./logs/textcnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=True,
    write_images=True, write_steps_per_second=False, update_freq='epoch',
    profile_batch=0, embeddings_freq=0, embeddings_metadata=None)

# 保存ckpt文件
ckpt_file_path = "./models/textcnnckpt/"
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_file_path, verbose=0, 
    monitor='val_loss', mode='min',
    save_freq='epoch', save_best_only=False,
    save_weights_only=True, 
    options=None, initial_value_threshold=None
)

# # 断点续训
# if os.path.exists(ckpt_file_path):
#     model.load_weights(ckpt_file_path)
#     # 若成功加载前面保存的参数，输出下列信息
#     print("checkpoint_loaded")
class_weight={
    0:1.0,
    1:1.0
}
history = model.fit(train_dataset, validation_data = val_dataset,
                    callbacks=[early_stopping, tensorboard_callback, ckpt_callback],
                    class_weight=class_weight, epochs=3, verbose=1)

# 模型评估和改进
# > tensorboard --logdir logs/mlp

test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print("=============")
print("在测试集的准确率: ", test_acc)
print("=============")

# 模型保存加载和部署
# # 模型加载和预测(ckpt)
# if os.path.exists(ckpt_file_path):
#     model.load_weights(ckpt_file_path)
#     test_input = np.expand_dims(test_texts[0],0)
#     pred = model.predict(test_input)
#     print(f"第一句文本的预测值: {np.argmax(pred)}, 概率: {np.max(pred)}")
#     print(f"第一句文本的真实值: {test_labels[0]}")

# 模型的保存(pb)
pb_file_path = './models/multiModel/textcnn/1'
model.save(pb_file_path, save_format='tf')

# # 模型加载和预测(pb)
# restored_saved_model=tf.keras.models.load_model(pb_file_path)
# test_input = np.expand_dims(test_texts[0],0)
# pred = restored_saved_model.predict(test_input)
# print(f"第一句文本的预测值: {np.argmax(pred)}, 概率: {np.max(pred)}")
# print(f"第一句文本的真实值: {test_labels[0]}")

# restored_saved_model.get_layer("dense_1").kernel # 查看模型参数

# 最后使用docker-tf-serving部署pb文件的模型，即可使用http在线访问预测
