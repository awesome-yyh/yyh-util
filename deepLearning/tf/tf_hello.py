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
    tf.config.experimental.set_memory_growth(gpu, True) # 根据需要来逐渐增长GPU内存
# tf.debugging.set_log_device_placement(True) # 打印使用的设备CPU/GPU


# 四则运算
print("------四则运算-------")
A = tf.constant([[1.0, 2.0], 
                 [3.0, 4.0]])
B = tf.constant([[5.0, 6.0], 
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
print(A.shape, A.dtype, A.ndim, tf.size(A).numpy(), A.numpy()) # (2, 2) <dtype: 'float32'> 2 4(2*2) [[1. 2.]
print(tf.reshape(A, [1, 1, -1])) # [[[1. 2. 3. 4.]]], shape=(1, 1, 4)
print(tf.reshape(A, [-1])) # [1. 2. 3. 4.] 展平张量

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

# layers.Dense
print("------layers.Dense-------")
class CustomDense(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomDense, self).__init__()
        self.units = units

    def build(self, input_shape): # 创建层的权重, 也可在__init__中创建
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    def get_config(self): # 使该层支持支持序列化，返回构造函数参数
        return {"units": self.units}
    
A = tf.constant([[1.0, 2.0], 
                 [3.0, 4.0]])

print('result 1:', tf.keras.layers.Dense(units=2, activation=None)(A))
print('result 2:', CustomDense(2)(A))

# layers.Dropout
print("------layers.Dropout-------")
class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

A = tf.constant([[1.0, 2.0], 
                 [3.0, 4.0]])

print('result 1:', tf.keras.layers.Dropout(0.5)(A))
print('result 2:', CustomDropout(0.5)(A))

# layers.Flatten
print("------layers.Flatten-------")

# layers.Conv2D
print("------layers.Conv2D-------")

# layers.MaxPooling2D
print("------layers.MaxPooling2D-------")

# layers.SimpleRNN
print("------layers.SimpleRNN-------")
class CustomRNN(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CustomRNN, self).__init__()
        self.units = units
        self.projection_1 = tf.keras.layers.Dense(units=units, activation="tanh")
        self.projection_2 = tf.keras.layers.Dense(units=units, activation="tanh")
        self.classifier = tf.keras.layers.Dense(1)

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = tf.stack(outputs, axis=1)
        return self.classifier(features)

A = tf.constant([[[1.0, 2.0], 
                 [3.0, 4.0]]])

print('result 1: ', tf.keras.layers.SimpleRNN(2)(A))
print('result 2: ', CustomRNN(2)(A))

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
class CustomAtt(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
    def call(self, query, key_list):
        scores = tf.matmul(query, key_list, transpose_b=True)
        # dk = tf.cast(tf.shape(key_list)[-1], dtype=tf.float32)
        # scores /= tf.math.sqrt(dk)
        distribution = tf.nn.softmax(scores)
        return tf.matmul(distribution, key_list)
    def get_config(self):
        return {}

query = tf.constant([[1.0, 2.0]])
key_list = tf.constant([[5.0, 6.0], 
                        [7.0, 8.0]])

print('result 1:', tf.keras.layers.Attention()([query, key_list]))
print('result 2:', CustomAtt()(query, key_list))

# 模型演示-线性回归
print("------模型演示-线性回归-------")
# 读取数据
# y = 2*x - 1
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0], dtype=float)

# 探索分析
# plt.plot(xs, ys, 'ro', label='Original data')
# plt.show()

# 数据清洗(缺失值、重复值、异常值、大小写、标点)

# 数据采样(搜集、合成、过采样、欠采样、阈值移动、loss加权、评价指标)

# 特征工程(数值、类别、时间、文本、图像)

# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
train_x, test_x, train_y, test_y = train_test_split(xs, ys, test_size=0.2, random_state=1)
print(train_x, test_x, train_y, test_y)

# 搭建模型
# 子类式API
class Linear(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(0.0)
        self.b = tf.Variable(0.0)
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
history = model.fit(train_x, train_y, epochs=10, callbacks=[early_stop], verbose=0)

# 模型评估和改进
test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
print("在测试集的准确率: ", test_acc)

# 模型预测
predict = model.predict(test_x).flatten()
print(f"输入的x: {test_x}, 真实值: {test_y}, 预测值: {predict}")
