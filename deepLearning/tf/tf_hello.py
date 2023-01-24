import os
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
print("Hub version: ", hub.__version__)
print("Tensorflow path: \n", tf.__path__)

print("Tensorflow is built with CUDA: ", tf.test.is_built_with_cuda())
print("CPU or GPU: \n", device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True) # 设置显存分配, 根据需要来逐渐增长GPU内存
# tf.debugging.set_log_device_placement(True) # 打印使用的设备CPU/GPU
os.environ['CUDA_VISIBLE_DEVICES']='2'


# 四则运算
print("------四则运算-------")
A = tf.Variable([[1.0, 2.0], 
                 [3.0, 4.0]])
B = tf.Variable([[5.0, 6.0], 
                 [7.0, 8.0]])
print(tf.add(A, 1)) # [[2,3],[4,5]] 所有元素+1
print(tf.add(A, B)) # [[6,8],[10,12]] 对应位相加
print(A+B) # 同上, [[6,8],[10,12]] 对应位相加

print(tf.matmul(A, B)) # [[19,22],[43,50]] 矩阵乘法，对应位相乘并相加
print(A @ B) # 同上, [[19,22],[43,50]] 矩阵乘法，对应位相乘并相加
print(tf.multiply(A, B)) # [[5,12],[21,32]] 对应位置相乘
print(A * B) # 同上, [[5,12],[21,32]] 对应位置相乘


# 查看矩阵A的形状、类型和值
print("------形状、类型和值-------")
print(A.shape, A.dtype, tf.size(A).numpy(), A.numpy()) # (2, 2) <dtype: 'float32'> 2 4(2*2) [[1. 2.]
print(tf.reshape(A, [1, 1, -1])) # [[[1. 2. 3. 4.]]], shape=(1, 1, 4)
print(tf.reshape(A, [-1])) # [1. 2. 3. 4.] 展平张量


# 元素访问和修改
print("------元素访问和修改-------")
T = tf.Variable([[1.0, 2.0], 
                 [3.0, 4.0]])
print(T[-1], T[0:1])
T[0,1].assign(99) # 原地操作（在原内存地址修改并生效）
print(T)


# 统计运算
print("------统计运算-------")
print(tf.reduce_max(A)) # 4, 最大值
print(tf.argmax(A)) # [1,1], 最大值的索引
print(tf.nn.softmax(A)) # [[0.26894143 0.7310586 ][0.26894143 0.7310586 ]]

print(tf.reduce_mean(A)) # 2.5 所有元素的平均值
print(tf.reduce_mean(A, axis=1)) # [1.5 3.5]

# 求导
print("------求导-------")
x = tf.Variable(3.)
with tf.GradientTape() as tape: # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x) # 同 y = x**2
    print(y) # 9
print(tape.gradient(y, x))   # 计算y关于x的导数, 6

# @tf.function
print("------@tf.function-------")
class F():
    def __init__(self):
        self._b = None
    @tf.function
    def __call__(self):
        a = tf.constant([[10, 10], [11., 1.]])
        x = tf.constant([[1., 0.], [0., 1.]])
        if self._b is None:
            self._b = tf.Variable(12.)
        y = tf.matmul(a, x) + self._b
        print("PRINT: ", y)
        tf.print("TF-PRINT: ", y)
        return y
f = F()
print(f())

# 模型演示-线性回归
# 即不加激活函数的全连接层
print("------模型演示-线性回归-------")
# 读取数据
# y = 2*x - 1
xs = np.array([i for i in range(10)], dtype=np.float32)
ys = np.array([2*i-1+np.random.rand(1) for i in xs], dtype=np.float32)

# # 探索分析
# plt.plot(xs, ys, 'ro', label='Original data')
# plt.show()

# 数据清洗(缺失值、重复值、异常值、大小写、标点)

# 数据采样(搜集、合成、过采样、欠采样、阈值移动、loss加权、评价指标)

# 特征工程(数值、类别、时间、文本、图像)

# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
train_x, test_x, train_y, test_y = train_test_split(xs, ys, test_size=0.2, random_state=1)
# print(train_x, test_x, train_y, test_y)

# 搭建模型
# 子类式API
class Linear(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(0.1)
        self.b = tf.Variable(0.1)
    def call(self, x):
        return self.w * x + self.b
    def build_graph(self, input_shape):
        input_ = tf.keras.layers.Input(shape=input_shape)
        return tf.keras.models.Model(inputs=[input_], outputs=self.call(input_))

# model = Linear()

# 函数式API
x = tf.keras.Input(shape=(1,))
y = tf.keras.layers.Dense(units=1, activation=None)(x)
model = tf.keras.Model(inputs=x, outputs=y)

# 查看模型结构
model.build((None,))
model.summary()
tf.keras.utils.plot_model(model, "deepLearning/tf/hellomodel.png", show_shapes=True)

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
