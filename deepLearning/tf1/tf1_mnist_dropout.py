from string import digits
from unittest import result
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
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

# 手写数字识别，分类模型
with tf.name_scope("inputs"):
    keep_prob =  tf.placeholder(tf.float32)
    xs = tf.placeholder(tf.float32, shape=[None, 64], name='x') # shape依次为输入维度、输出维度，None代表不做要求，都可以
    ys = tf.placeholder(tf.float32, shape=[None, 10], name='y') # 输出种类

l1 = add_layer(xs, 64, 100, activation_fun=tf.nn.relu, layer_name="layer1") # 隐含层
prediction = add_layer(l1, 100, 10, activation_fun=tf.nn.softmax, layer_name="layer2") # 输出层

with tf.name_scope("loss"):
    # 损失函数：softmax交叉熵，先写真实值，再写预测值, 
    # 如果有batch的话，它的大小就是[batchsize，num_classes]，
    # 单样本的话，大小就是num_classes
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(0.02).minimize(loss) # 优化器：梯度下降，学习率是0.5

# data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
# 划分训练集和测试集, random_state是随机数的种子，不填则每次都不同
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# train
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables()) # 初始化所有变量
    tf.train.Saver().save(sess, "D:\\ml\\net\\mymodel.ckpt") # checkpoint, 保存模型
    # 导入模型使用时需要同样的shape和dtype
    # tf.train.Saver().restore(sess, "D:\\ml\\net\\mymodel.ckpt")
    
    merged = tf.summary.merge_all() 
    train_write = tf.summary.FileWriter("D:\\ml\\logs\\train", sess.graph)
    test_write = tf.summary.FileWriter("D:\\ml\\logs\\test", sess.graph)
    
    for step in range(1001):
        sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob:0.6})
        
        if step % 20 == 0:
            print(f"step: {step}/1000, \
                acc:{compute_acc(X_test, y_test)} \
                train loss: {sess.run(loss, feed_dict={xs:X_train, ys:y_train, keep_prob:1})}, \
                test loss: {sess.run(loss, feed_dict={xs:X_test, ys:y_test, keep_prob:1})}") # 输出精度和loss
            train_result = sess.run(merged, feed_dict={xs:X_train, ys:y_train, keep_prob:1})
            train_write.add_summary(train_result, step)
            text_result = sess.run(merged, feed_dict={xs:X_test, ys:y_test, keep_prob:1})
            test_write.add_summary(text_result, step)
