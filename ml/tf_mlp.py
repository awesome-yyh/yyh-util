import datetime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt


# 读取数据
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0
test_images = test_images/255.0

# 数据清洗

# 特征工程

# 搭建模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss = tf.keras.losses.sparse_categorical_crossentropy, 
              metrics=['accuracy'])

# 训练模型
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    verbose=1,
    patience=10, # 每10步检查一下是否提升
    mode='max', # monitor是准确率时max，是损失时min
    restore_best_weights=True)

log_dir="./logs/mlp/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=True,
    write_images=True, write_steps_per_second=False, update_freq='epoch',
    profile_batch=0, embeddings_freq=0, embeddings_metadata=None)

history = model.fit(training_images, training_labels, 
                    validation_data = (test_images, test_labels),
                    callbacks=[early_stopping, tensorboard_callback],
                    epochs=6, verbose=2)

# 画图分析过拟合欠拟合，并改进模型
model.summary()
# > tensorboard --logdir logs/mlp

# 保存模型

# 模型加载和预测
classifications = model.predict(test_images)

print(f"第一张图片的预测值: {np.argmax(classifications[0])}, 概率: {np.max(classifications[0])}")
print(f"第一张图片的真实值: {test_labels[0]}")
