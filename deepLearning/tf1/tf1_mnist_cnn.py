import re
from unittest import result
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def add_layer(inputs, in_size, out_size, activation_fun=None, layer_name="layer"):
    with tf.name_scope(layer_name):
        with tf.name_scope("weight"):
            Weight = tf.Variable(tf.random_normal([in_size, out_size], dtype=tf.float32),
                            name = "Weight") # 随机生成指定形状的浮点数，默认均值是0，方差是1
            tf.summary.histogram('Weight', Weight)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size], dtype=tf.float32)+0.1,
                                name = "biases")
            tf.summary.histogram('biases', biases)
        with tf.name_scope("wx_b"):
            outputs = tf.matmul(inputs, Weight) + biases
            outputs = tf.nn.dropout(outputs, keep_prob)
        if activation_fun:
            outputs = activation_fun(outputs) # 有默认name
        
        tf.summary.histogram('outputs', outputs)
        return outputs

def compute_acc(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1}) # 预测值，是10个概率值，所以下一步取最大值
    currect_pre = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1)) # 预测值和真实值比较
    acc = tf.reduce_mean(tf.cast(currect_pre, tf.float32)) # 统计这组数据的正确率， cast是改变数据类型
    result = sess.run(acc, feed_dict={xs:v_xs, ys:v_ys, keep_prob:1})
    return result

def Weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W): # x是输入值
    # strides=[1,x_move,y_move,1] 第一个和第四个都是要等于1，第二项是x方向的跨度，第三项是y方向的跨度
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    # ksize=[1,heigh,width,1] 池化窗口的大小
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# 搭建模型
with tf.name_scope("inputs"):
    keep_prob =  tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, shape=[None, 28*28], name='x') # shape依次为输入维度、输出维度，None代表不做要求，都可以
    ys = tf.placeholder(tf.float32, shape=[None, 10], name='y') # 输出种类
    x_img = tf.reshape(xs, [-1,28,28,1]) # 最后一个1是颜色

# 第1个卷积层
W_conv1 = Weight_variable([5,5,1,32]) # patch是5*5，通道数是1，卷积核（过滤器）数是32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1) # out size=8*8*32
h_pool1 = max_pool_2x2(h_conv1) # out size = 4*4*32, 一次跨2步，所以8/2

# 第2个卷积层
W_conv2 = Weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # out size=8*8*64
h_pool2 = max_pool_2x2(h_conv2) # out size = 2*2*32, 一次跨2步，所以8/2

# func1层
W_fc1 = Weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# func2层
W_fc2 = Weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

# prediction = add_layer(xs, 28*28, 10, activation_fun=tf.nn.softmax, layer_name="layer1")

with tf.name_scope("loss"):
    # 损失函数：softmax交叉熵，先写真实值，再写预测值, 
    # 如果有batch的话，它的大小就是[batchsize，num_classes]，
    # 单样本的话，大小就是num_classes
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss) # 优化器：梯度下降，学习率是0.5

# data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # 如果没有数据集会自动下载到这个文件

# train
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables()) # 初始化所有变量
    merged = tf.summary.merge_all() 
    write = tf.summary.FileWriter("D:\\ml\\logs", sess.graph)
    
    for step in range(1001):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:1})
        
        if step % 20 == 0:
            print(f"{step}/1000, \
                acc: {compute_acc(mnist.test.images, mnist.test.labels)}, \
                loss: {sess.run(loss, feed_dict={xs:mnist.test.images, ys:mnist.test.labels, keep_prob:1})}") # 输出精度和loss
            result = sess.run(merged, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:1})
            write.add_summary(result, step)
