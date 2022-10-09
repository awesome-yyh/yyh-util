import re
from unittest import result
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from tensorflow.examples.tutorials.mnist import input_data
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
        
        if activation_fun:
            outputs = activation_fun(outputs) # 有默认name
        
        tf.summary.histogram('outputs', outputs)
        return outputs

def compute_acc(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs:v_xs}) # 预测值，是10个概率值，所以下一步取最大值
    currect_pre = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1)) # 预测值和真实值比较
    acc = tf.reduce_mean(tf.cast(currect_pre, tf.float32)) # 统计这组数据的正确率， cast是改变数据类型
    result = sess.run(acc, feed_dict={xs:v_xs, ys:v_ys})
    return result

# 训练数据
mnist = tf.keras.datasets.mnist
# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# training_images = training_images/255.0
# test_images = test_images/255.0
# data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # 如果没有数据集会自动下载到这个文件

# 搭建模型
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, shape=[None, 28*28], name='x') # shape依次为输入维度、输出维度，None代表不做要求，都可以
    ys = tf.placeholder(tf.float32, shape=[None, 10], name='y') # 输出种类

prediction = add_layer(xs, 28*28, 10, activation_fun=tf.nn.softmax, layer_name="layer1")

with tf.name_scope("loss"):
    # 损失函数：softmax交叉熵，先写真实值，再写预测值, 
    # 如果有batch的话，它的大小就是[batchsize，num_classes]，
    # 单样本的话，大小就是num_classes
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(0.02).minimize(loss) # 优化器：梯度下降，学习率是0.5


# 训练模型
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables()) # 初始化所有变量
    merged = tf.summary.merge_all() 
    write = tf.summary.FileWriter("D:\\ml\\logs", sess.graph)
    
    for step in range(1001):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
        
        if step % 20 == 0:
            print(f"{step}/1000, \
                acc: {compute_acc(mnist.test.images, mnist.test.labels)}, \
                test loss: {sess.run(loss, feed_dict={xs:mnist.test.images, ys:mnist.test.labels})}") # 输出精度和loss
            result = sess.run(merged, feed_dict={xs:batch_xs, ys:batch_ys})
            write.add_summary(result, step)
