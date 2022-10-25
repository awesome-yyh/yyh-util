import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 自动随机切分训练数据和测试数据
from sklearn import preprocessing


print("Tensorflow version: ", tf.__version__)
print("Tensorflow is built with CUDA: ", tf.test.is_built_with_cuda())
print("Tensorflow path: \n", tf.__path__)
print("CPU or GPU: \n", device_lib.list_local_devices())

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.add(A, B)
D = tf.matmul(A, B)
print(C)
print(D)
# 查看矩阵A的形状、类型和值
print(D.shape, D.dtype, D.numpy())

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print(y, y_grad)


# 线性回归
# 读取数据
# y = 2*x - 1
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# 探索分析
# plt.plot(xs, ys, 'ro', label='Original data')
# plt.show()

# 数据清洗

# 数据变换
# xs = preprocessing.StandardScaler().fit_transform(xs.reshape(-1, 1))
# ys = preprocessing.StandardScaler().fit_transform(ys.reshape(-1, 1))

# 特征工程

# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
train_x, test_x, train_y, test_y = train_test_split(xs, ys, test_size=0.2, random_state=1)


# 搭建模型
x = tf.keras.Input(shape=(1,))
y = tf.keras.layers.Dense(units=1, activation=None,
                        kernel_initializer=tf.zeros_initializer(),
                        bias_initializer=tf.zeros_initializer())(x)
model = tf.keras.Model(inputs=x, outputs=y)

# 查看模型结构
# model.build((None,))
model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

model.compile(loss='mse', optimizer='sgd', metrics=['mae'])

# 训练模型
early_stop=tf.keras.callbacks.EarlyStopping(monitor='loss',
                                            patience=20) # 每20步检查一下是否提升
history = model.fit(train_x, train_y, epochs=10, callbacks=[early_stop], verbose=2)

# 模型评估和改进
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print("在测试集的准确率: ", test_acc)

# 模型预测
predict = model.predict(test_x).flatten()
print(f"输入的x: {test_x}, 真实值: {test_y}, 预测值: {predict}")
