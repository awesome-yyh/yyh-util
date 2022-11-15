import datetime
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf


# 读取数据
# !wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
# !unzip jena_climate_2009_2016.csv.zip
fname = os.path.join("data/jena_climate_2009_2016.csv")
with open(fname) as f:
    data = f.read()
lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print(header, lines[0])
print(len(lines))

temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]
    raw_data[i, :] = values[:]
print(temperature.shape, raw_data.shape)
print(temperature[0], raw_data[0])

# 数据探索分析
# plt.plot(range(len(temperature)), temperature)
# plt.show()

# 数据清洗(缺失值、重复值、异常值、大小写、标点)

# 数据采样(搜集、合成、过采样、欠采样、阈值移动、loss加权、评价指标)

# 特征工程(数值、文本、类别、时间)
# raw_data = preprocessing.StandardScaler().fit_transform(raw_data)

# 划分训练集、验证集、测试集
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)

sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

val_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)

test_dataset = tf.keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)

# 搭建模型
inputs = tf.keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = tf.keras.layers.SimpleRNN(32, recurrent_dropout=0.25)(inputs)
# x = tf.keras.layers.LSTM(32, recurrent_dropout=0.25)(inputs)
x = tf.keras.layers.GRU(16)(inputs)
# x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))(inputs)
# x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(inputs)
# x = tf.keras.layers.Dropout(0.5)(x)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)

# 查看模型结构
# model.build((None,))
model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss = "mse", metrics=["mae"])

# 训练模型
# early_stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',
    verbose=1,
    patience=10, # 每10步检查一下是否提升
    mode='max', # monitor是准确率时max，是损失时min
    restore_best_weights=True)

# tensorboard
log_dir="./logs/rnn/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir, histogram_freq=1, write_graph=True,
    write_images=True, write_steps_per_second=False, update_freq='epoch',
    profile_batch=0, embeddings_freq=0, embeddings_metadata=None)

# 保存ckpt文件
ckpt_file_path = "./models/rnnckpt/"
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_file_path, verbose=0, 
    monitor='val_loss', mode='min',
    save_freq='epoch', save_best_only=False,
    save_weights_only=True, 
    options=None, initial_value_threshold=None
)

# 断点续训
# if os.path.exists(ckpt_file_path):
#     model.load_weights(ckpt_file_path)
#     # 若成功加载前面保存的参数，输出下列信息
#     print("checkpoint_loaded")

history = model.fit(train_dataset, 
                    validation_data = val_dataset,
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
acc = history_dict["mae"]
val_acc = history_dict["val_mae"]
plt.plot(epochs, acc, "bo", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

test_loss, test_mae = model.evaluate(test_dataset, verbose=2)
print("在测试集的准确率: ", test_mae)

# 模型保存加载和部署
# 模型加载和预测(ckpt)
if os.path.exists(ckpt_file_path):
    model.load_weights(ckpt_file_path)
    classifications = model.predict(test_dataset)
    print(f"第一个预测值: {np.argmax(classifications[0])}, 概率: {np.max(classifications[0])}")

# 模型的保存(pb)
pb_file_path = './models/multiModel/rnn/1'
model.save(pb_file_path, save_format='tf')
# 或 tf.keras.models.save_model(model, pb_file_path)

# 模型加载和预测(pb)
restored_saved_model=tf.keras.models.load_model(pb_file_path)
test_input = np.expand_dims(list(test_dataset.unbatch())[0][0],0)
pred = restored_saved_model.predict(test_input) # 模型预测
print(f"第一个预测值: {np.argmax(pred)}")

# restored_saved_model.get_layer("dense_1").kernel # 查看模型参数

# 最后使用docker-tf-serving部署pb文件的模型，即可使用http在线访问预测
