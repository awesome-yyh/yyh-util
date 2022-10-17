import datetime
import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
# from pyspark.sql import SparkSession


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

# 读取数据
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

# 数据清洗

# 特征工程

# TFRecord
# spark-submit --class xxx.xx.xx.mainObject  --master local[2]   /opt/xxx.jar
# spark-submit --class xxx.xx.xx.mainObject  --master local[2]   /opt/xxx.jar
# df = spark.read.options(delimiter=',') \
#   .csv('/Users/yaheyang/sample.csv', header=False)
# df.write.mode("overwrite").format("tfrecord").option("recordType", "Example").save("/Users/yaheyang/ss") # spark 写tfrecord

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
                    validation_data = (test_images, test_labels),
                    callbacks=[early_stopping, tensorboard_callback, cp_callback],
                    epochs=3, verbose=2)

# 画图分析过拟合欠拟合，并改进模型
model.summary()
# > tensorboard --logdir logs/mlp

# 模型加载和预测(ckpt)
model.load_weights(ckpt_file_path)
classifications = model.predict(test_images)
print(f"第一张图片的预测值: {np.argmax(classifications[0])}, 概率: {np.max(classifications[0])}")
print(f"第一张图片的真实值: {test_labels[0]}")

# 模型的保存(pb)
pb_file_path = './models/multiModel/cnn/100000'
tf.keras.models.save_model(model, pb_file_path)

# 模型加载和预测(pb)
restored_saved_model=tf.keras.models.load_model(pb_file_path)
test_input = np.expand_dims(test_images[0],0)
pred = restored_saved_model.predict(test_input) # 模型预测
print(f"第一张图片的预测值: {np.argmax(pred)}")
print(f"第一张图片的真实值: {test_labels[0]}")

# restored_saved_model.get_layer("dense_1").kernel # 查看模型参数
