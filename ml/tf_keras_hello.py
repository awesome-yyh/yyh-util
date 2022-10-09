import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 自动随机切分训练数据和测试数据


print("Tensorflow version: ", tf.__version__)
print("Tensorflow is built with CUDA: ", tf.test.is_built_with_cuda())
print("Tensorflow path: \n", tf.__path__)
print("CPU or GPU: \n", device_lib.list_local_devices())


# 训练数据
# y = 2*x - 1
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
# plt.plot(xs, ys, 'ro', label='Original data')
# plt.show()

# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
train_x, test_x, train_y, test_y = train_test_split(xs, ys, test_size=0.2, random_state=1)


# 搭建模型
x = tf.keras.Input((1,))
y = tf.keras.layers.Dense(units=1)(x)
model = tf.keras.Model(x, y)

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(1, input_shape = [1]))
model.compile(loss='mse', optimizer='sgd', metrics=['mae'])

# 训练模型
early_stop=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=20) # 每20步检查一下是否提升
model.fit(train_x, train_y, epochs=5, callbacks=[early_stop], verbose=2)


# 模型结构、预测及评估
model.summary()

predict = model.predict(test_x).flatten()
print(f"输入的x: {test_x}, 真实值: {test_y}, 预测值: {predict}")

[loss,mae] = model.evaluate(test_x, test_y, verbose=2)
print(f"对于测试数据, loss: {loss}, mae:{mae}")
