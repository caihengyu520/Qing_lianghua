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
from _load_mnist_train import load_mnist_train
from _load_mnist_test import load_mnist_test
from _Goods import Goods
from _Data import Data
from _load_model_conv import load_model_conv
from _compute_accuracy import compute_accuracy
from _compute_accuracy_ import compute_accuracy_
from _load_model_3_conv import load_model_3_conv
from _load_model_3_1_conv import load_model_3_1_conv
from _weights_conv import weights_conv
random.seed(200)

flags=tf.flags
FLAGS = flags.FLAGS

## Dataset
flags.DEFINE_string('data_base_path', 'fashion_mnist', 'fashion_mnist or mnist_dataset')
flags.DEFINE_integer('test_batch_size', 200, 'test_batch_size is the number of every time')
flags.DEFINE_string('optimizer_way', 'Gradient_descent', 'fine_tuning optimizer way [Gradient_descent,RRMSPropOptimizer]')

decoder2_1=None
prediction_1=None

X_2 = None
tst_2 = None
yhat_2 = None
decoder2_2=None


train_data_size=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
# test_data_size=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
test_data_size=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
# test_data_size=[1,1,1,1,1,1,1,1,1,1]
# batch_data_size=[1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
train_x,train_y=load_mnist_train(train_data_size,FLAGS.data_base_path)
test_x,test_y=load_mnist_test(test_data_size,FLAGS.data_base_path)
# batch_data_,batch_label_=load_test(batch_data_size)
test_x_=test_x
test_y_=test_y
batch_size=FLAGS.test_batch_size
index_list = [i for i in range(test_y.shape[0])]
random.shuffle(index_list)
print("chu shi shujuji suoyin:",index_list)


class Jichen():
    def __init__(self):
        pass

# class Inputdata(threading.Thread):
#     def __init__(self,data,i_p_condition,goods_1):
#         threading.Thread.__init__(self)
#         self.i_p_condition=i_p_condition
#         self.goods_1=goods_1
#         self.data=data
#     def run(self):
#         i_p=self.i_p_condition
#         goods_1=self.goods_1
#         data=self.data
#         while True:
#             if goods_1.empty():
#                 data.set_data()
#                 time.sleep(1)
#                 goods_1.add()
#                 print("输入数据线程开始睡3秒---------------------")
#                 time.sleep(3)
#                 print("输入数据线程睡3秒成功---------------------")
#                 goods_1.sub()


#生产者类，想当于只进行推断，将推断结果传递给微调模型
class Producer(threading.Thread):
    def __init__(self,data_2,condition,resources, goods,goods_1,test_x,test_y,weights_1,count_=1, sleeptime=2):  # sleeptime=2
        threading.Thread.__init__(self)
        self.cond = condition
        self.goods = goods
        self.goods_1=goods_1
        self.sleeptime = sleeptime
        self.test_x=test_x
        self.test_y=test_y
        self.count_=count_
        self.resources=resources
        self.data_2=data_2
        self.weights_1=weights_1

    def run(self):
        cond = self.cond
        goods = self.goods
        resources=self.resources
        X_1=None
        tst_1=None
        jishuqi=1
        graph=tf.Graph()
        sess1=tf.Session(graph=graph)
        while True:
            cond.acquire()
            while True:
                self.data_2.set_data()
                train_data=self.data_2.train_x
                train_label=self.data_2.train_y
                if goods.empty():
                    goods.add()
                    flag=1
                    if jishuqi==1:
                        flag=0
                    else:
                        flag=1
                        sess1.close()
                        tf.reset_default_graph()
                        graph = tf.Graph()
                        sess1 = tf.Session(graph=graph)
                    jishuqi+=1
                    resources.acquire()
                    decoder2_1,X_1,tst_1=load_model_conv(sess1,train_data,train_label,self.weights_1,flag,self.test_x,self.test_y)
                    resources.release()
                result_1, precison_1 = compute_accuracy(train_data, train_label, sess1, X_1,
                                                        tst_1, decoder2_1)
                print('test_produce_result_1:', result_1)
                print('test_produce_precision_1:', precison_1)
                result_1_, precison_1 = compute_accuracy(self.test_x, self.test_y, sess1, X_1,
                                                         tst_1, decoder2_1)
                print('final_produce_result_1:', result_1_)
                print('final_produce_precision_1:', precison_1)
                # cond.notifyAll()  # 唤醒所有等待的线程--》其实就是唤醒消费者进程
                print("通知消费者",time.time())
                # cond.release()  # 解锁资源
                time.sleep(self.sleeptime)
                if result_1<=0.5:
                    # data_2.get_data(data.train_data,data.train_label)
                    self.goods_1.add(1)
                    cond.release()
                    time.sleep(1)
                    #在该位置给data_2传递数据
                    break
                elif result_1<=0.7:
                    # data_2.get_data(data.train_data,data.train_label)
                    self.goods_1.add(2)
                    cond.release()
                    time.sleep(1)
                    #在该位置给data_2传递数据
                    break
                if result_1<=0.9:
                    # data_2.get_data(data.train_data,data.train_label)
                    self.goods_1.add(3)
                    cond.release()
                    time.sleep(1)
                    #在该位置给data_2传递数据
                    break
                else:
                    continue

# 消费者类，相当与对模型进行微调，并将微调之后的结果进行保存
class Consumer(threading.Thread):
    def __init__(self,data_2,condition,resources, goods,goods_1, test_x,test_y,weights_1,count_,optimizer_way="Gradient_descent",sleeptime=1):  # sleeptime=1
        threading.Thread.__init__(self)
        self.cond = condition
        self.goods = goods
        self.sleeptime = sleeptime
        self.train_x=train_x
        self.train_y=train_y
        self.count_=count_
        self.resources=resources
        self.data_2=data_2
        self.test_x=test_x
        self.test_y=test_y
        self.weights_1=weights_1
        self.goods_1=goods_1
        self.optimizer_way=optimizer_way
    def run(self):
        cond = self.cond
        goods = self.goods
        resources=self.resources
        X_3=None
        tst_3=None
        decoder2_3=None
        while True:
            time.sleep(self.sleeptime)
            print("消费者准备锁住资源",time.time())
            #锁住资源，用于使模型微调时接收器模型暂无法进行数据的读取和推断
            cond.acquire()
            print("消费者成功锁住资源")
            train_data=self.data_2.train_data
            train_label=self.data_2.train_label
            train_len=self.data_2.len
            #在该位置接收data_2的数据并进行操作
            time.sleep(1)
            graph=tf.Graph()
            sess3=tf.Session(graph=graph)
            #count计数器用来表示是第一次加载模型还是之前已经加载过模型，涉及到图恢复后是否需要在重新定义某些节点和操作
            if self.count_ == 1:
                X_3, tst_3, decoder2_3= load_model_3_conv(
                    sess3, train_data, train_label,train_len,resources,goods,self.weights_1,self.goods_1,self.optimizer_way)
            else:
                sess3,X_3, tst_3, decoder2_3 = load_model_3_1_conv(
                    sess3,train_data, train_label,train_len,resources,goods,self.weights_1,self.goods_1,self.optimizer_way)
            self.count_ += 1
            result_1, precison_1 = compute_accuracy_(train_data, train_label, sess3, X_3, tst_3, decoder2_3)
            print('test_consumer_result_1:', result_1)
            print('test_consumer_precision_1:', precison_1)
            result_1_, precison_1 = compute_accuracy_(self.test_x, self.test_y, sess3,X_3,tst_3,decoder2_3)
            print('final_consumer_result_1:', result_1_)
            print('final_consumer_precision_1:', precison_1)
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
weights_1=weights_conv()
data_2=Data(test_x,test_y,batch_size)
#当采用权重文件时需要此资源锁，但当使用权重结构体保存时，就可以忽略此结构体锁
condition = threading.Condition()
resources=threading.Condition()

pro = Producer(data_2,condition,resources,g,g_1,test_x_,test_y_,weights_1,count_=1)
pro.start()

con = Consumer(data_2,condition,resources,g,g_1,test_x_,test_y_,weights_1,count_=2,optimizer_way=FLAGS.optimizer_way)
con.start()
#开启两个线程用于模拟模型的推断和微调