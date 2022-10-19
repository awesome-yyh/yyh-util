import datetime
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# from pyspark.sql import SparkSession


# 方式一：使用spark对数据读取、清洗、特征工程、划分，最后存储为TFRecord
# # SparkSession是整个spark应用的起点
# # appName 是在yarn管理界面查看的应用名称
# spark = SparkSession.builder \
#       .master("local") \
#       .appName("yyhALS") \
#       .config("spark.submit.deployMode","client") \
#       .config("spark.port.maxRetries", "100") \
#       .config("spark.sql.broadcastTimeout", "1200") \
#       .config("spark.yarn.queue", "root.search") \
#       .enableHiveSupport() \
#       .getOrCreate()

# TFRecord
# spark-submit --class xxx.xx.xx.mainObject  --master local[2]   /opt/xxx.jar
# spark-submit --class xxx.xx.xx.mainObject  --master local[2]   /opt/xxx.jar
# df = spark.read.options(delimiter=',') \
#   .csv('/Users/yaheyang/sample.csv', header=False)
# df.write.mode("overwrite").format("tfrecord").option("recordType", "Example").save("/Users/yaheyang/ss") # spark 写tfrecord

# 方式二：使用numpy、pandas、matplotlib对数据读取、清洗、特征工程、划分，最后存储为TFRecord
# 数据读取
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# 探索性数据分析

# 数据清洗

# 数据变换
training_images = training_images/255.0
test_images = test_images/255.0

# 特征工程

# 划分训练集、验证集、测试集
training_images, val_images, training_labels, val_labels = train_test_split(
    training_images, training_labels, test_size=0.2, random_state=1, stratify=training_labels)

# 存储为TFRecord
tfrecords_path = "./data/mlptfrecord/train.tfrecords"
with tf.io.TFRecordWriter(tfrecords_path) as file_writer:
    for training_image, training_label in zip(training_images, training_labels):
        record_bytes = tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[training_image.astype(np.float64).tobytes()])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[training_label])),
        })).SerializeToString()
        file_writer.write(record_bytes)

# 将tfrecords数据解析出来
def decode_tfrecords(example):
    # 定义Feature结构，告诉解码器每个Feature的类型是什么
    feature_description = {                        
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    #按照feature_description解码
    feature_dict = tf.io.parse_single_example(example, feature_description)
    #由bytes码转化为tf.int32
    image=(tf.io.decode_raw(feature_dict['image'],out_type=tf.float64))
    #修改维度为编码前
    image=tf.reshape(image,[60000,28,28])

    return image,feature_dict['label']

#读取tfrecords文件
def read_tfrecords():
    #读取文件
    dataset = tf.data.TFRecordDataset(tfrecords_path)  # 读取 TFRecord 文件
    dataset = dataset.map(decode_tfrecords)  # 解析数据


# 搭建模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss = tf.keras.losses.sparse_categorical_crossentropy, 
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
log_dir="./logs/mlp/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=True,
    write_images=True, write_steps_per_second=False, update_freq='epoch',
    profile_batch=0, embeddings_freq=0, embeddings_metadata=None)

# 保存ckpt文件
ckpt_file_path = "./models/mlpckpt/"
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_file_path, verbose=0, 
    monitor='val_loss', mode='min',
    save_freq='epoch', save_best_only=False,
    save_weights_only=True, 
    options=None, initial_value_threshold=None
)

# 断点续训
if os.path.exists(ckpt_file_path):
    model.load_weights(ckpt_file_path)
    # 若成功加载前面保存的参数，输出下列信息
    print("checkpoint_loaded")

history = model.fit(training_images, training_labels, 
                    validation_data = (val_images, val_labels),
                    callbacks=[early_stopping, tensorboard_callback, cp_callback],
                    epochs=3, verbose=2)

# 画图（训练误差、验证误差），分析过拟合欠拟合，并调整超参数改进模型
model.summary()
# > tensorboard --logdir logs/mlp

# 模型评估（测试误差）
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("在测试集的准确率: ", test_acc)

# 模型加载和预测(ckpt)
if os.path.exists(ckpt_file_path):
    model.load_weights(ckpt_file_path)
    classifications = model.predict(test_images)
    print(f"第一张图片的预测值: {np.argmax(classifications[0])}, 概率: {np.max(classifications[0])}")
    print(f"第一张图片的真实值: {test_labels[0]}")

# 模型的保存(pb)
pb_file_path = './models/multiModel/mlp/1'
tf.keras.models.save_model(model, pb_file_path)

# 模型加载和预测(pb)
restored_saved_model=tf.keras.models.load_model(pb_file_path)
test_input = np.expand_dims(test_images[0],0)
pred = restored_saved_model.predict(test_input) # 模型预测
print(f"第一张图片的预测值: {np.argmax(pred)}")
print(f"第一张图片的真实值: {test_labels[0]}")

# restored_saved_model.get_layer("dense_1").kernel # 查看模型参数

# 最后使用docker-tf-serving部署pb文件的模型，即可使用http在线访问预测
