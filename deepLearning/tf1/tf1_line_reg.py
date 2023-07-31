import tensorflow as tf
import numpy as np


## line reg model
xs = tf.placeholder(tf.float32, shape=[100,], name='x') # shape依次为输入维度、输出维度
ys = tf.placeholder(tf.float32, shape=[100,], name='y')

Weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0, dtype=tf.float32), 
                    name = "Weight") # 指定形状、最小值、最大值的均匀分布中取出随机数
biases = tf.Variable(tf.random_uniform([1], -1.0, 1.0), 
                    name = "biases")

y = Weight * xs + biases
loss = tf.losses.mean_squared_error(y, ys) # 损失函数：均方差
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 优化器：梯度下降，学习率是0.5

## data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 2 + 3

## train
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables()) # 初始化所有变量
    print(f"初始参数: {sess.run(Weight)}, {sess.run(biases)}")
    for step in range(201):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if step % 20 == 0:
            print(f"{step}/200, {sess.run(Weight)}, {sess.run(biases)}")
