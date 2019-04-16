# 卷积网络的训练数据为MNIST(28*28灰度单色图像)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from _load_data import load_data
from _load_test import load_test
# from _load_mnist_test_false import load_mnist_test_false
from _next_batch_ import next_batch_
import pandas as pd
import scipy.misc
# from _heatmap import heatmap

random.seed(200)

train_epochs = 10000    # 训练轮数
batch_update=1
batch_size   = 100     # 随机出去数据大小
display_step = 100     # 显示训练结果的间隔
learning_rate= 0.0001  # 学习效率
drop_prob    = 0.5     # 正则化,丢弃比例
fch_nodes    = 512     # 全连接隐藏层神经元的个数
start=0

# 网络模型需要的一些辅助函数
# 权重初始化(卷积核初始化)
# tf.truncated_normal()不同于tf.random_normal(),返回的值中不会偏离均值两倍的标准差
# 参数shpae为一个列表对象,例如[5, 5, 1, 32]对应
# 5,5 表示卷积核的大小, 1代表通道channel,对彩色图片做卷积是3,单色灰度为1
# 最后一个数字32,卷积核的个数,(也就是卷基层提取的特征数量)
#   显式声明数据类型,切记
def weight_init(shape,name):
    weights = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(weights,name=name)

# 偏置的初始化
def biases_init(shape,name):
    biases = tf.random_normal(shape,dtype=tf.float32)
    return tf.Variable(biases,name=name)

# 随机选取mini_batch
# def get_random_batchdata(n_samples, batchsize):
#     start_index = np.random.randint(0, n_samples - batchsize)
#     return (start_index, start_index + batchsize)

# 全连接层权重初始化函数xavier
def xavier_init(layer1, layer2,name ,constant = 1):
    Min = -constant * np.sqrt(6.0 / (layer1 + layer2))
    Max = constant * np.sqrt(6.0 / (layer1 + layer2))
    return tf.Variable(tf.random_uniform((layer1, layer2), minval = Min, maxval = Max, dtype = tf.float32),name=name)

# 卷积
def conv2d(x, w,name):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME',name=name)
# 源码的位置在tensorflow/python/ops下nn_impl.py和nn_ops.py
# 这个函数接收两个参数,x 是图像的像素, w 是卷积核
# x 张量的维度[batch, height, width, channels]
# w 卷积核的维度[height, width, channels, channels_multiplier]
# tf.nn.conv2d()是一个二维卷积函数,
# stirdes 是卷积核移动的步长,4个1表示,在x张量维度的四个参数上移动步长
# padding 参数'SAME',表示对原始输入像素进行填充,卷积后映射的2D图像与原图大小相等
# 填充,是指在原图像素值矩阵周围填充0像素点
# 如果不进行填充,假设 原图为 32x32 的图像,卷积和大小为 5x5 ,卷积后映射图像大小 为 28x28

# 池化
def max_pool_2x2(x,name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name=name)


# train_data_size=[1,1,1,1,1,1,1,1,1,1]
# test_data_size=[1,1,1,1,1,1,1,1,1,1]
# valid_data_size=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]

train_data_size=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
test_data_size=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
valid_data_size=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]

train_x,train_y=load_data(train_data_size)
test_x,test_y=load_test(test_data_size)
valid_x,valid_y=load_test(valid_data_size)
test_y_=np.array(test_y)
test_x_=np.array(test_x)
index_list = [i for i in range(train_y.shape[0])]
random.shuffle(index_list)
print(index_list)
# x 是手写图像的像素值,y 是图像对应的标签
x = tf.placeholder(tf.float32, [None, 784],name="x")
y = tf.placeholder(tf.float32, [None, 10],name="y")
# 把灰度图像一维向量,转换为28x28二维结构
x_image = tf.reshape(x, [-1, 28, 28, 1])
# -1表示任意数量的样本数,大小为28x28深度为一的张量
# 可以忽略(其实是用深度为28的,28x1的张量,来表示28x28深度为1的张量)



w_conv1 = weight_init([5, 5, 1, 16],name="w_conv1")        # 5x5,深度为1,16个
b_conv1 = biases_init([16],name="b_conv1")
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1,name="conv1") + b_conv1)    # 输出张量的尺寸:28x28x16
h_pool1 = max_pool_2x2(h_conv1,name="pool1")                                   # 池化后张量尺寸:14x14x16
# h_pool1 , 14x14的16个特征图


w_conv2 = weight_init([5, 5, 16, 32],name="w_conv2")                             # 5x5,深度为16,32个
b_conv2 = biases_init([32],name="b_conv2")
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2,name="conv2") + b_conv2)    # 输出张量的尺寸:14x14x32
h_pool2 = max_pool_2x2(h_conv2,name="pool2")                                   # 池化后张量尺寸:7x7x32
# h_pool2 , 7x7的32个特征图

# h_pool2是一个7x7x32的tensor,将其转换为一个一维的向量
h_fpool2 = tf.reshape(h_pool2, [-1, 7*7*32],name="h_fpool2")
# 全连接层,隐藏层节点为512个
# 权重初始化
w_fc1 = xavier_init(7*7*32, fch_nodes,name="w_fc1")
b_fc1 = biases_init([fch_nodes],name="b_fc1")
h_fc1 = tf.nn.relu(tf.matmul(h_fpool2, w_fc1) + b_fc1,name="fc1")

# 全连接隐藏层/输出层
# 为了防止网络出现过拟合的情况,对全连接隐藏层进行 Dropout(正则化)处理,在训练过程中随机的丢弃部分
# 节点的数据来防止过拟合.Dropout同把节点数据设置为0来丢弃一些特征值,仅在训练过程中,
# 预测的时候,仍使用全数据特征
# 传入丢弃节点数据的比例
#keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=drop_prob)

# 隐藏层与输出层权重初始化
w_fc2 = xavier_init(fch_nodes, 10,name="w_out")
b_fc2 = biases_init([10],name="b_out")

# 未激活的输出
y_ = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2,name="out")
# 激活后的输出
y_out = tf.nn.softmax(y_,name="out_final")

# #tensorboard xiangguan
# tf.summary.histogram("wd1", w_conv1)
# tf.summary.histogram("wd2", w_conv2)
# tf.summary.histogram("bd1", b_conv1)
# tf.summary.histogram("bd2", b_conv2)
# tf.summary.histogram("wfc1", w_fc1)
# tf.summary.histogram("bfc1", b_fc1)
# tf.summary.histogram("wout", w_fc2)
# tf.summary.histogram("bout", b_fc2)

# #tensorboard卷积核
# x_min=tf.reduce_min(w_conv1)
# x_max=tf.reduce_max(w_conv1)
# kernel_0_1=(w_conv1-x_min)/(x_max-x_min)
# kernel_transposed=tf.transpose(kernel_0_1,[3,0,1,2])
#
# x_min1=tf.reduce_min(w_conv2)
# x_max1=tf.reduce_max(w_conv2)
# kernel_0_11=(w_conv2-x_min1)/(x_max1-x_min1)
# kernel_transposed1=tf.transpose(kernel_0_11,[3,0,1,2])

# tf.summary.image("cov1/filters",kernel_transposed,max_outputs=5)
# tf.summary.image("cov2/filters",kernel_transposed1,max_outputs=5)

# 交叉熵代价函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_out), reduction_indices = [1]),name="cross_entropy")

# tensorflow自带一个计算交叉熵的方法
# 输入没有进行非线性激活的输出值 和 对应真实标签
#cross_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))

# 优化器选择Adam(有多个选择)
optimizer = tf.train.AdamOptimizer(learning_rate,name="optimizer").minimize(cross_entropy)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate,name="optimizer").minimize(cross_entropy)
# 准确率
# 每个样本的预测结果是一个(1,10)的vector
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_out, 1),name="correct_prediction")
# tf.cast把bool值转换为浮点数
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")

# tf.summary.scalar("cost",cross_entropy)
# tf.summary.scalar("accuracy",accuracy)
#
# merged=tf.summary.merge_all()
# log_path="mnist_logs/logs7"

saver=tf.train.Saver(max_to_keep=1)

# 会话
with tf.Session() as sess:
    if os.path.exists('AE1/fashion_model/checkpoint'):
        model_file = tf.train.latest_checkpoint('AE1/fashion_model/')
        saver.restore(sess, model_file)
    else:
        # 全局变量进行初始化的Operation
        init = tf.global_variables_initializer()
        sess.run(init)
    # writer = tf.summary.FileWriter(log_path, sess.graph)
    # init = tf.global_variables_initializer()
    # sess.run(init)
    step=1
    Cost = []
    Accuracy = []
    for i in range(train_epochs):
        for j in range(batch_update):
            start,index_list,batch_x,batch_y = next_batch_(train_x,train_y, batch_size,start,index_list)
            _, cost, accu = sess.run([ optimizer, cross_entropy,accuracy], feed_dict={x:batch_x, y:batch_y})
            # summary = sess.run(merged, feed_dict={x: batch_x, y: batch_y})
            # writer.add_summary(summary, step)
            step+=1
            Cost.append(cost)
            Accuracy.append(accu)
        if i % display_step ==0:
            for j in range(2):
                cost, accu = sess.run([cross_entropy,accuracy], feed_dict={x:valid_x, y:valid_y})
                print ('Epoch : %d ,  Cost : %.7f'%(i+1, cost))
                print('Epoch : %d ,  accuracy : %.7f' % (i + 1, accu))
                saver.save(sess,"AE1/fashion_model/model.ckpt")
    print('training finished')
    _, cost, accu = sess.run([optimizer, cross_entropy, accuracy], feed_dict={x: test_x, y: test_y})
    print("final cost:",cost)
    print("final accu:",accu)

