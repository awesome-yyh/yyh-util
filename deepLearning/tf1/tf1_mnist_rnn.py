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
        
        if activation_fun:
            outputs = activation_fun(outputs) # 有默认name
        
        tf.summary.histogram('outputs', outputs)
        return outputs

# def compute_acc(v_xs, v_ys):
#     global prediction
#     y_pre = sess.run(prediction, feed_dict={xs:v_xs}) # 预测值，是10个概率值，所以下一步取最大值
#     currect_pre = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1)) # 预测值和真实值比较
#     acc = tf.reduce_mean(tf.cast(currect_pre, tf.float32)) # 统计这组数据的正确率， cast是改变数据类型
#     result = sess.run(acc, feed_dict={xs:v_xs, ys:v_ys})
#     return result

# 训练参数
lr = 0.001
training_iters = 1000
batch_size = 128

n_inputs = 28 # mnist 28*28, 28列
n_steps = 28 # 28行
n_hidden_unis = 128 # 隐藏层神经元的个数
n_classes = 10 # mnist 0-9

def RNN(X, weight, bias):
    # X: (128batch, 28step, 28inputs)
    X = tf.reshape(X, [-1, n_inputs])
    X_in = tf.matmul(X, weight['in'])+bias['in']
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_unis])
    
    # LSTM 
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    
    result = tf.matmul(states[1], weight['out'])+bias['out']
    
    return result



# 手写数字识别，分类模型
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, shape=[None, n_steps*n_inputs], name='x') # shape依次为输入维度、输出维度，None代表不做要求，都可以
    ys = tf.placeholder(tf.float32, shape=[None, n_classes], name='y') # 输出种类

weights = {
    # 28*128
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),
    # 128*10
    'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes])),
}

bias = {
    # 128,
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unis,])),
    # 10,
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,])),
}

prediction = RNN(xs, weights, bias)
# prediction = add_layer(xs, 28*28, 10, activation_fun=tf.nn.softmax, layer_name="layer1")

with tf.name_scope("loss"):
    # 损失函数：softmax交叉熵，先写真实值，再写预测值, 
    # 如果有batch的话，它的大小就是[batchsize，num_classes]，
    # 单样本的话，大小就是num_classes
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
    tf.summary.scalar("loss", loss)
    currect_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1)) # 预测值和真实值比较
    acc = tf.reduce_mean(tf.cast(currect_pred, tf.float32)) # 统计这组数据的正确率， cast是改变数据类型

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(0.02).minimize(loss) # 优化器：梯度下降，学习率是0.5

# data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # 如果没有数据集会自动下载到这个文件

# train
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables()) # 初始化所有变量
    merged = tf.summary.merge_all() 
    write = tf.summary.FileWriter("D:\\ml\\logs", sess.graph)
    
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_step], feed_dict={xs:batch_xs, ys:batch_ys})
        
        if step % 20 == 0:
            print(f"{step}/1000, \
                acc: {sess.run(acc, feed_dict = {xs: batch_xs, ys: batch_ys})}, \
                test loss: {sess.run(loss, feed_dict={xs:mnist.test.images, ys:mnist.test.labels})}") # 输出精度和loss
            result = sess.run(merged, feed_dict={xs:batch_xs, ys:batch_ys})
            write.add_summary(result, step)
        step += 1
