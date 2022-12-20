import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 自动随机切分训练数据和测试数据
from sklearn import preprocessing
import tensorflow_hub as hub


# tf的基本信息
print("------tf的基本信息-------")
print("Tensorflow version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Tensorflow is built with CUDA: ", tf.test.is_built_with_cuda())
print("Tensorflow path: \n", tf.__path__)
print("CPU or GPU: \n", device_lib.list_local_devices())
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
print("Hub version: ", hub.__version__)


# 四则运算
print("------四则运算-------")
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
print(tf.add(A, B)) # [[6,8],[10,12]]
print(tf.matmul(A, B)) # [[19,22],[43,50]]

# 查看矩阵A的形状、类型和值
print(A.shape, A.dtype, A.numpy()) # (2, 2) <dtype: 'int32'> [[1 2][3 4]]

# 统计运算
print("------统计运算-------")

# 求导
print("------求导-------")
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape: # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
    print(y) # 9
print(tape.gradient(y, x))   # 计算y关于x的导数, 6

# layers.Flatten
print("------layers.Flatten-------")

# layers.Dense
print("------layers.Dense-------")

# layers.Dropout
print("------layers.Dropout-------")

# layers.Conv2D
print("------layers.Conv2D-------")

# layers.MaxPooling2D
print("------layers.MaxPooling2D-------")

# layers.SimpleRNN
print("------layers.SimpleRNN-------")

# layers.LSTM
print("------layers.LSTM-------")

# layers.GRU
print("------layers.GRU-------")

# layers.Bidirectional
print("------layers.Bidirectional-------")
# x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16))(inputs)
# x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16))(inputs)


# layers.Attention
print("------layers.Attention-------")
query = tf.convert_to_tensor(np.asarray([[[1., 1., 1., 3.]]]))

key_list = tf.convert_to_tensor(np.asarray([[[1., 1., 2., 4.], [4., 1., 1., 3.], [1., 1., 2., 1.]]
                                            ]))

print('query shape:', query.shape)
print('key shape:', key_list.shape)

query_value_attention_seq = tf.keras.layers.Attention()([query, key_list])
print('result 1:',query_value_attention_seq)

scores = tf.matmul(query, key_list, transpose_b=True)
distribution = tf.nn.softmax(scores)
print('distribution: ', distribution)
result = tf.matmul(distribution, key_list)
print('result 2:',query_value_attention_seq)

# 模型演示-线性回归
print("------模型演示-线性回归-------")
# 读取数据
# y = 2*x - 1
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# 探索分析
# plt.plot(xs, ys, 'ro', label='Original data')
# plt.show()

# 数据清洗(缺失值、重复值、异常值、大小写、标点)

# 数据采样(搜集、合成、过采样、欠采样、阈值移动、loss加权、评价指标)

# 特征工程(数值、类别、时间、文本、图像)
# xs = preprocessing.StandardScaler().fit_transform(xs.reshape(-1, 1))
# ys = preprocessing.StandardScaler().fit_transform(ys.reshape(-1, 1))

# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
train_x, test_x, train_y, test_y = train_test_split(xs, ys, test_size=0.2, random_state=1)
print(train_x, test_x, train_y, test_y)

# 搭建模型
x = tf.keras.Input(shape=(1,))
y = tf.keras.layers.Dense(units=1, activation=None,
                        kernel_initializer=tf.zeros_initializer(),
                        bias_initializer=tf.zeros_initializer())(x)
model = tf.keras.Model(inputs=x, outputs=y)

# 查看模型结构
model.build((None,))
model.summary()
# tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

model.compile(loss='mse', optimizer='sgd', metrics=['mae'])

# 训练模型
early_stop=tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=20) # 每20步检查一下是否提升
history = model.fit(train_x, train_y, epochs=200, callbacks=[early_stop], verbose=2)

# 模型评估和改进
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print("在测试集的准确率: ", test_acc)

# 模型预测
predict = model.predict(test_x).flatten()
print(f"输入的x: {test_x}, 真实值: {test_y}, 预测值: {predict}")
