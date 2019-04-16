#thread_5线程和thread_4线程具备同样功能


"""
本实例中，开启三个线程分别是Inputdata，Producer，Consumer线程。其中输入数据线程通过g_1产品与生产者通信，
并且通过data_1中的数据传递给生产者作为更新集。然后生产者通过g_1产品以及c和resource信号量来控制消费者的行为，
让消费者实现模型微调功能，然后将微调后的模型进行保存。之后生产者再次读取更新后的模型并将其作为当前的模型。
"""

"""
使用 Condition 类来完成，由于它也可以像锁机制那样用，所以它也有 acquire 方法和 release 方法，而且它还有
wait， notify， notifyAll 方法。
"""

import threading
import time,random
import tensorflow as tf
# import pandas as pd
# import numpy as np
# from sklearn.metrics import confusion_matrix
# import os
# from functools import reduce
# from operator import mul
# import shutil
# from _next_batch import next_batch
from _load_data import load_data
from _load_test import load_test
from _save_model import save_model
from _Goods import Goods
from _Data import Data
from _load_model import load_model
from _compute_accuracy import compute_accuracy
from _compute_accuracy_ import compute_accuracy_
from _load_model_3 import load_model_3
from _load_model_3_1 import load_model_3_1
random.seed(100)


def test_1(test_x,test_y,sess1):
    """
        Convert data to Numpy array which has a shape of (-1, 41, 41, 41, 3).
        Test a single axample.
        Arg:
                txtdata: Array in C.
        Returns:
            The normal of a face.
    """
    global X_1, tst_1, yhat_1,decoder2_1
    start_time=time.time()
    output,output1= sess1.run([yhat_1,decoder2_1], feed_dict={X_1: test_x, tst_1: test_y})  # (100, 3)
    duration_time=time.time()-start_time
    return output,output1,duration_time


def test_2(test_x,test_y,sess2):
    """
        Convert data to Numpy array which has a shape of (-1, 41, 41, 41, 3).
        Test a single axample.
        Arg:
                txtdata: Array in C.
        Returns:
            The normal of a face.
    """
    global X_1, tst_1, yhat_2,decoder2_2
    start_time=time.time()
    output,output1= sess2.run([yhat_2,decoder2_2], feed_dict={X_2: test_x, tst_2: test_y})  # (100, 3)
    duration_time=time.time()-start_time
    return output,output1,duration_time


# X_1 = None
# tst_1 = None
# yhat_1 = None
decoder2_1=None
prediction_1=None

X_2 = None
tst_2 = None
yhat_2 = None
decoder2_2=None

# X_3 = None
# tst_3 = None
# yhat_3 = None
# decoder2_3=None
# encoder2_3=None


train_data_size=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
# test_data_size=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
test_data_size=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
# test_data_size=[1,1,1,1,1,1,1,1,1,1]
train_x,train_y=load_data(train_data_size)
test_x,test_y=load_test(test_data_size)
test_x_=test_x
test_y_=test_y
batch_size=100
# update_size=1

# class Goods:  # 产品类
#     def __init__(self):
#         self.count = 0
#
#     def add(self, num=1):
#         self.count += num
#
#     def sub(self):
#         if self.count >= 0:
#             self.count =0
#
#     def empty(self):
#         return self.count <= 0
#
#     def notempty(self):
#         return self.count>0




class Jichen():
    def __init__(self):
        pass

class Inputdata(threading.Thread):
    def __init__(self,data,i_p_condition,goods_1):
        threading.Thread.__init__(self)
        self.i_p_condition=i_p_condition
        self.goods_1=goods_1
        self.data=data
    def run(self):
        i_p=self.i_p_condition
        goods_1=self.goods_1
        data=self.data
        while True:
            if goods_1.empty():
                data.set_data()
                time.sleep(1)
                goods_1.add()
                print("输入数据线程开始睡3秒---------------------")
                time.sleep(3)
                print("输入数据线程睡3秒成功---------------------")
                goods_1.sub()



class Producer(threading.Thread):  # 生产者类
    def __init__(self,data,data_2,i_p_condition,condition,resources, goods,goods_1,test_x,test_y,count_=1, sleeptime=2):  # sleeptime=2
        threading.Thread.__init__(self)
        self.i_p_condition=i_p_condition
        self.cond = condition
        self.goods = goods
        self.goods_1=goods_1
        self.sleeptime = sleeptime
        self.test_x=test_x
        self.test_y=test_y
        self.count_=count_
        self.resources=resources
        self.data=data
        self.data_2=data_2
    def run(self):
        i_p=self.i_p_condition
        cond = self.cond
        goods = self.goods
        goods_1=self.goods_1
        resources=self.resources
        data=self.data
        data_2=self.data_2
        X_1=None
        tst_1=None
        yhat_1=None
        jishuqi=1
        graph=tf.Graph()
        sess1=tf.Session(graph=graph)
        decoder2=None
        p_c_count=0
        while True:
            # cond.acquire()  # 锁住资源
            # i_p.acquire()
            # i_p.wait()
            # i_p.release()
            cond.acquire()
            while True:
                p_c_count += 1
                # while goods_1.empty():
                #     continue
                data_2.set_data()
                if goods.empty():
                    goods.add()
                    if jishuqi==1:
                        pass
                    else:
                        sess1.close()
                        tf.reset_default_graph()
                        graph = tf.Graph()
                        sess1 = tf.Session(graph=graph)
                    jishuqi+=1
                    resources.acquire()
                    decoder2,X_1,tst_1,yhat_1=load_model(sess1,data_2.get_train_data(),data_2.get_train_label())
                    resources.release()
                result_1, precison_1 = compute_accuracy(data_2.get_train_data(), data_2.get_train_label(), sess1,X_1,tst_1,yhat_1, decoder2)
                print('produce_result_1:', result_1)
                print('produce_precision_1:', precison_1)
                result_1, precison_1 = compute_accuracy(test_x, test_y, sess1, X_1,
                                                        tst_1, yhat_1, decoder2)
                print('produce_result_1:', result_1)
                print('produce_precision_1:', precison_1)
                # cond.notifyAll()  # 唤醒所有等待的线程--》其实就是唤醒消费者进程
                print("通知消费者",time.time())
                # cond.release()  # 解锁资源
                time.sleep(self.sleeptime)
                if result_1<=0.9:
                    # data_2.get_data(data.train_data,data.train_label)
                    cond.release()
                    time.sleep(1)
                    #在该位置给data_2传递数据
                    break
                else:
                    continue


class Consumer(threading.Thread):  # 消费者类
    def __init__(self,data,condition,resources, goods, test_x,test_y,count_,sleeptime=1):  # sleeptime=1
        threading.Thread.__init__(self)
        self.cond = condition
        self.goods = goods
        self.sleeptime = sleeptime
        self.train_x=train_x
        self.train_y=train_y
        self.count_=count_
        self.resources=resources
        self.data=data
        self.test_x=test_x
        self.test_y=test_y
    def run(self):
        cond = self.cond
        goods = self.goods
        resources=self.resources
        data=self.data
        X_3=None
        tst_3=None
        yhat_3=None
        decoder2_3=None
        encoder2_3=None
        while True:
            time.sleep(self.sleeptime)
            print("消费者准备锁住资源",time.time())
            # cond.acquire()  # 锁住资源
            # while goods.empty():  # 如无产品则让线程等待
            #     cond.wait()
            cond.acquire()
            print("消费者成功锁住资源")
            train_data=data.get_train_data()
            train_label=data.get_train_label()
            train_len=data.get_train_len()
            #在该位置接收data_2的数据并进行操作
            time.sleep(1)
            graph=tf.Graph()
            sess3=tf.Session(graph=graph)

            if self.count_ == 1:
                X_3, tst_3, yhat_3, decoder2_3,e1_value, e2_value, e3_value, e4_value, e5_value, e6_value, e7_value, e8_value = load_model_3(
                    sess3,X_3,tst_3,yhat_3,decoder2_3,encoder2_3, train_data, train_label,train_len,resources,goods)
            else:
                X_3, tst_3, yhat_3, decoder2_3,e1_value, e2_value, e3_value, e4_value, e5_value, e6_value, e7_value, e8_value = load_model_3_1(
                    sess3,X_3,tst_3,yhat_3,decoder2_3,encoder2_3, train_data, train_label,train_len,resources,goods)
            self.count_ += 1
            result_1, precison_1 = compute_accuracy_(train_data, train_label, sess3, X_3, tst_3, yhat_3, decoder2_3)
            print('consumer_result_1:', result_1)
            print('consumer_precision_1:', precison_1)
            result_1, precison_1 = compute_accuracy_(test_x, test_y, sess3,X_3,tst_3,yhat_3,decoder2_3)
            print('consumer_result_1:', result_1)
            print('consumer_precision_1:', precison_1)
            sess3.close()
            cond.release()
            print("产品数量:", goods.count, "消费者线程")
            print(time.time())
            # time.sleep(5)
            # print(time.time())
            print("消费者睡了")
            # cond.release()  # 解锁资源

g = Goods()
g_1=Goods()
data_1=Data(test_x,test_y,batch_size)
data_2=Data(test_x,test_y,batch_size)
c = threading.Condition()
resources=threading.Condition()
i_p=threading.Condition()

pro = Producer(data_1,data_2,i_p,c,resources,g,g_1,test_x_,test_y_,count_=1)
pro.start()

con = Consumer(data_2,c,resources,g,test_x,test_y,count_=2)
con.start()

# input_data=Inputdata(data_1,i_p,g_1)
# input_data.start()