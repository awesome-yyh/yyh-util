import datetime
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# 使用cnn对mnist手写数字识别(function模式)

# 读取数据
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# 数据探索分析

# 数据清洗(缺失值、重复值、异常值、大小写、标点)

# 类别不均衡(搜集、合成、过采样、欠采样、阈值移动、loss加权、更改评价指标)

# 特征工程(数值、文本、类别、时间)
training_images=training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

training_images=training_images / 255.0
test_images=test_images/255.0

# 划分训练集、验证集、测试集
training_images, val_images, training_labels, val_labels = train_test_split(
    training_images, training_labels, test_size=0.2, random_state=1, stratify=training_labels)


# 搭建模型
inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# 查看模型结构
# model.build((None,))
model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

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
log_dir="./logs/cnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=True,
    write_images=True, write_steps_per_second=False, update_freq='epoch',
    profile_batch=0, embeddings_freq=0, embeddings_metadata=None)

# 保存ckpt文件
ckpt_file_path = "./models/cnnckpt/"
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
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
                    callbacks=[early_stopping, tensorboard_callback, ckpt_callback],
                    batch_size = 128, epochs=3, verbose=2)

# 模型评估和改进
# > tensorboard --logdir logs/mlp
import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.subplot(121)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(122)
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("在测试集的准确率: ", test_acc)

# 模型保存加载和部署
# 模型加载和预测(ckpt)
if os.path.exists(ckpt_file_path):
    model.load_weights(ckpt_file_path)
    classifications = model.predict(test_images)
    print(f"第一张图片的预测值: {np.argmax(classifications[0])}, 概率: {np.max(classifications[0])}")
    print(f"第一张图片的真实值: {test_labels[0]}")

# 模型的保存(pb)
pb_file_path = './models/multiModel/cnn/1'
model.save(pb_file_path, save_format='tf')
# 或 tf.keras.models.save_model(model, pb_file_path)

# 模型加载和预测(pb)
restored_saved_model=tf.keras.models.load_model(pb_file_path)
test_input = np.expand_dims(test_images[0],0)
pred = restored_saved_model.predict(test_input) # 模型预测
print(f"第一张图片的预测值: {np.argmax(pred)}")
print(f"第一张图片的真实值: {test_labels[0]}")

# restored_saved_model.get_layer("dense_1").kernel # 查看模型参数

# 最后使用docker-tf-serving部署pb文件的模型，即可使用http在线访问预测

import matplotlib.pyplot as plt
digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
