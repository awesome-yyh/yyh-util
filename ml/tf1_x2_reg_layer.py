import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_fun=None):
    Weight = tf.Variable(tf.random_normal([in_size, out_size], dtype=tf.float32),
                    name = "Weight") # 随机生成指定形状的浮点数，默认均值是0，方差是1
    biases = tf.Variable(tf.zeros([1, out_size], dtype=tf.float32)+0.1,
                        name = "biases")
    outputs = tf.matmul(inputs, Weight) + biases
    
    if activation_fun:
        outputs = activation_fun(outputs)
    
    return outputs


# 一元二次方程的回归模型
xs = tf.placeholder(tf.float32, shape=[None, 1], name='x_input') # shape依次为输入维度、输出维度，None代表不做要求，都可以
ys = tf.placeholder(tf.float32, shape=[None, 1], name='y_input')

l1 = add_layer(xs, 1, 10, activation_fun = tf.nn.relu)
prediction = add_layer(l1, 10, 1)

loss = tf.losses.mean_squared_error(prediction, ys) # 损失函数：均方差
train_step = tf.train.AdamOptimizer(0.1).minimize(loss) # 优化器：梯度下降，学习率是0.5

## data
x_data = np.linspace(-1,1,300)[:,np.newaxis].astype(np.float32)
noise = np.random.normal(0,0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) + 3 + noise

## plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

## train
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables()) # 初始化所有变量
    print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
    # print(f"初始参数: {sess.run(Weight)}, {sess.run(biases)}")
    for step in range(201):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if step % 20 == 0:
            print(f"{step}/200, {sess.run(loss, feed_dict={xs:x_data, ys:y_data})}")
            try:
                ax.lines.remove(line[0])
            except Exception:
                pass
            predict = sess.run(prediction, feed_dict={xs:x_data}) # 与x_data有关, 结果是所有y值，而非一个值
            line = ax.plot(x_data, predict, 'r-', lw=1)
            plt.pause(0.5)
            # print(f"{step}/200, {sess.run(Weight)}, {sess.run(biases)}")
plt.pause(0) # 暂停
