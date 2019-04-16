import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix
import time
import os
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul
import collections
random.seed(100)

# 导入MNIST数据
def next_batch(train_data,train_target,batch_size):
    #内置采样函数，保证采取到不同的行索引
    index=random.sample(range(0,train_target.shape[0]-1),batch_size)
    batch_data=train_data.iloc[index]
    batch_target=train_target.iloc[index]
    return batch_data,batch_target

#获得总的参数量
def get_num_params():
    num_params=0
    for variable in tf.trainable_variables():
        shape=variable.get_shape()
        num_params+=reduce(mul,[dim.value for dim in shape],1)
    return num_params

def load_data(data_size):
    i=0
    train_temp = pd.DataFrame()
    train_temp_y = pd.Series()
    for size in data_size:
        train_temp = pd.concat([train_temp, pd.DataFrame(data=pd.read_csv('fashion_mnist/train/leibie_' + str(i)))],ignore_index=True)
        train_temp_y = pd.concat([train_temp_y, pd.Series(data=[i for z in range(6000)])], ignore_index=True)
        i += 1
        # print('train_temp:',train_temp)
        # print('train_temp_y',train_temp_y)
    # index = []
    # k=0
    # length=0
    # for size in data_size:
    #     length+=int(size*6000)
    #     while(len(index)<length):
    #         x=random.randint(k*6000,(k+1)*6000-1)
    #         if x not in index:
    #             index.append(x)
    #     k+=1
    print(train_temp)
    index=[]
    k=0
    for size in data_size:
        index_=random.sample(range(k*6000,(k+1)*6000),int(size*6000))
        for x in index_:
            index.append(x)
        k+=1
    print(len(index))
    index_=np.reshape(index,len(index))
    # print(collections.Counter(index_))
    data=train_temp.iloc[index_]
    target=train_temp_y.iloc[index_]
    train_data_y = pd.get_dummies(target)
    data/=256
    print(data)
    print(train_data_y)
    return data,train_data_y

# def load_test_data(data_size):
#     i=0
#     train_temp = pd.DataFrame()
#     train_temp_y = pd.Series()
#     for size in data_size:
#         train_temp = pd.concat([train_temp, pd.DataFrame(data=pd.read_csv('fashion_mnist/test/leibie_' + str(i)))],ignore_index=True)
#         train_temp_y = pd.concat([train_temp_y, pd.Series(data=[i for z in range(1000)])], ignore_index=True)
#         i += 1
#     index=[]
#     k=0
#     for size in data_size:
#         index.append(random.sample(range(k*1000,(k+1)*1000-1),int(size*1000)))
#         k+=1
#     index_=np.reshape(index,[1000])
#     data=train_temp.iloc[index_]
#     target=train_temp_y.iloc[index_]
#     train_data_y = pd.get_dummies(target)
#     data/=256
#     print(data)
#     print(train_data_y)
#     return data,train_data_y

def load_test_data(data_size):
    i=0
    train_temp = pd.DataFrame()
    train_temp_y = pd.Series()
    for size in data_size:
        # train_temp.append(pd.DataFrame(data=pd.read_csv('fashion_mnist/test/leibie_' + str(i))))
        train_temp = pd.concat([train_temp, pd.DataFrame(data=pd.read_csv('fashion_mnist/test/leibie_' + str(i)))],ignore_index=True)
        train_temp_y = pd.concat([train_temp_y, pd.Series(data=[i for z in range(1000)])], ignore_index=True)
        i += 1
    train_y = pd.get_dummies(train_temp_y)
    train_temp/=256
    return train_temp, train_y

def compute_accuracy(v_xs,v_ys,sess):
    #prediction 变为全剧变量
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    #预测值每行是10列，tf.argmax(数据，axis），相等为1，不想等为0
    # print(sess.run(ys,feed_dict={ys:v_ys}))
    # print(sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1}))
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    # 计算平均值，即计算准确率

    #引入混淆矩阵为了计算各个类别的预测准确度
    confusion_matrix1 = confusion_matrix(tf.argmax(sess.run(ys,feed_dict={ys:v_ys}),1).eval(), tf.argmax(sess.run(prediction,feed_dict={xs:v_xs}),1).eval())
    # print(confusion_matrix1)
    line=np.zeros(10)
    precision=np.zeros(10)
    for i in range(10):
        for j in range(10):
            line[i] += confusion_matrix1[i][j]
    for i in range(10):
        precision[i] = float(confusion_matrix1[i][i]) / line[i]
    print('precision:',precision)
    # print('W_fc2:',W_fc2.eval())

    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    # 运行我们的accuracy这一步
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result,precision
def var_filter(var_list):
    fileter_keywards=['weight_fully','bias_fully','weight_out','bias_out']
    for var in var_list:
        for layer in range(2):
            kw=fileter_keywards[layer]
            if kw in var.name:
                yield var
                break
            else:
                continue


train_data_size=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
test_data_size=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
train_x,train_y=load_data(train_data_size)
test_x,test_y=load_test_data(test_data_size)
xs = tf.placeholder(tf.float32,[None,784],name='xs')
ys = tf.placeholder("float",[None,10],name="ys")
learning_rate = 0.01
training_epochs = 10000
batch_size = 256
display_step = 1
examples_to_show = 10
n_input = 784

# tf Graph input (only pictures)
# X = tf.placeholder("float", [None, n_input])

# 用字典的方式存储各隐藏层的参数
n_hidden_1 = 256  # 第一编码层神经元个数
n_hidden_2 = 128  # 第二编码层神经元个数
n_hidden_3= 1000   # 全连接层神经元个数
# 权重和偏置的变化在编码层和解码层顺序是相逆的
# 权重参数矩阵维度是每层的 输入*输出，偏置参数维度取决于输出层的单元数
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name='encoder_h1'),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name='encoder_h2'),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]),name='decoder_h1'),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, 10]),name='decoder_h2'),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1]),name='encoder_b1'),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2]),name='encoder_b2'),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1]),name='decoder_b1'),
    'decoder_b2': tf.Variable(tf.random_normal([10]),name='decoder_b2'),
}

# 每一层结构都是 xW + b
# 构建编码器
def encoder(x):
    # with tf.name_scope("encoder"):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']),name='encoder1')
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']),name='encoder2')
    return layer_2


# 构建解码器
def decoder(x):
    # with tf.name_scope("decoder"):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']),name='decoder1')
    layer_2 = tf.nn.softmax(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']),name='decoder2')
    return layer_2


encoder_op = encoder(xs)
decoder_op = decoder(encoder_op)


# 预测
prediction = decoder_op
y_true = ys

# 定义代价函数和优化器
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-8,1.0)),reduction_indices=[1]),name='cross_entropy')

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

saver=tf.train.Saver(max_to_keep=1)


with tf.Session() as sess:
    max_accuracy=0.0
    writer = tf.summary.FileWriter('AE1/logs/', sess.graph)
    print(get_num_params())
    f = open('AE1/acc12.txt', 'w')
    f.write('train_data_size: ' + str(train_data_size) + '\n')
    f.write('test_data_size: ' + str(test_data_size) + '\n')
    # if os.path.exists('tmp_3/checkpoint'):
    #     model_file=tf.train.latest_checkpoint('tmp_3/')
    #     saver.restore(sess,model_file)
    # else:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(training_epochs):
        batch_xs, batch_ys = next_batch(train_x, train_y, batch_size=100)
        start_time=time.time()
        _, loss = sess.run([train_step, cross_entropy], feed_dict={xs: batch_xs, ys: batch_ys})
        duration_time=time.time()-start_time
        if i % 50 == 0:
            print('train time is:',duration_time)
            batch_xs, batch_ys = next_batch(train_x, train_y, batch_size=100)
            print('loss:', loss)
            start_time=time.time()
            result, precision = compute_accuracy(test_x, test_y, sess)
            duration_time=time.time()-start_time
            f.write(str(i) + ', val_acc: ' + str(result) + '\n')
            f.write(str(i) + ', precision: ' + str(precision) + '\n')
            saver.save(sess, 'AE1/model/model.ckpt')
            print('accuracy:', result)
    f.close()
