#恢复模型，从已经微调过的模型开始恢复
import tensorflow as tf
import time
import shutil
from _next_batch import next_batch
from _get_var_list import get_var_list
from _get_num_params import get_num_params
from _next_batch_ import next_batch_
import random

#修改过后的tensorflow模型保存，这里的保存是以结构体形式保存， 避免保存时增加图节点是模型体积变大
def load_model_3_1_conv(sess3,test_x,test_y,test_len,resources,goods,weights_1,good_1,optimizer_way):
    """
        Loading the pre-trained model and parameters.
    """

    # global X_3, tst_3, yhat_3,decoder2_3,encoder2_3
    with sess3.as_default():
        with sess3.graph.as_default():
            print("------------------------------load_model_3_1")
            modelpath = r'AE1/fashion_model2/'
            saver = tf.train.import_meta_graph(modelpath + 'model.ckpt.meta')
            saver.restore(sess3, tf.train.latest_checkpoint(modelpath))
            graph1 = tf.get_default_graph()
            X_3 = graph1.get_tensor_by_name("x:0")
            tst_3 = graph1.get_tensor_by_name("y:0")
            yhat_3 = graph1.get_tensor_by_name("cross_entropy:0")
            cost=graph1.get_tensor_by_name("cost:0")
            decoder2_3=graph1.get_tensor_by_name("out_final:0")
            encoder_h11 = graph1.get_tensor_by_name("w_conv1:0")
            encoder_b11 = graph1.get_tensor_by_name("b_conv1:0")
            encoder_h21 = graph1.get_tensor_by_name("w_conv2:0")
            encoder_b21 = graph1.get_tensor_by_name("b_conv2:0")
            decoder_h11 = graph1.get_tensor_by_name("w_fc1:0")
            decoder_b11 = graph1.get_tensor_by_name("b_fc1:0")
            decoder_h21 = graph1.get_tensor_by_name("w_out:0")
            decoder_b21 = graph1.get_tensor_by_name("b_out:0")
            saver_=tf.train.Saver(max_to_keep=1)
            targ_list=['w_fc1','b_fc1','w_out','b_out']
            var_list=list(tf.trainable_variables())
            trg=list(get_var_list(targ_list,var_list))
            learning_rate=0.0
            if good_1.count==1:
                learning_rate=1e-5
            elif good_1.count==2:
                learning_rate=1e-5
            else:
                learning_rate=1e-5
            good_1.sub()               #实现学习率信号清楚
            print("learning_rate is:",learning_rate)
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tst_3 * tf.log(decoder2_3), reduction_indices=[1]))
            # result=tf.nn.softmax(tf.add(tf.matmul(encoder2_3,weights_out),bias_out))
            train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
            if(optimizer_way=="Gradient_descent"):
                train_step_=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,var_list=trg)
            batch_size=50
            update_1=0
            update_size=50
            update_2=20
            show=20
            start=0
            index_list = [i for i in range(test_len)]
            random.shuffle(index_list)
            print("index:",index_list)

            # for i in range(update_1):
            #     index, batch_x, batch_y = fine_update(test_x, test_y, update_size, test_len, index)
            #     _, loss_1 = sess3.run([train_step, cost], feed_dict={X_3: batch_x, tst_3: batch_y})
            #     if (i % show == 0):
            #         index, batch_x, batch_y = fine_update(test_x, test_y, update_size, test_len, index)
            #         start_time = time.time()
            #         _, loss_train = sess3.run([train_step, cost], feed_dict={X_3: batch_x, tst_3: batch_y})
            #         duration = time.time() - start_time
            #         # print('train iter  {}    time is  {}'.format(i, duration))
            #         # print('parameters numbers is {}'.format(get_num_params()))
            #         # print('[Train] Step: %d, loss: %4.5f' % (i, loss_train))
            # # print("decoder_h1:",sess3.run(decoder_h1))
            # # print("encoder_h1:",sess3.run(encoder_h1))
            # print('test_consumer_result_1:', result_1)
            # print('test_consumer_precision_1:', precison_1)
            sum_time=0.0
            for i in range(update_2):
                # start,index_list,batch_x, batch_y = next_batch_(test_x, test_y, update_size,start,index_list)
                batch_x, batch_y = next_batch(test_x, test_y, update_size)
                start_time=time.time()
                _, loss_1 = sess3.run([train_step_, yhat_3], feed_dict={X_3: batch_x, tst_3: batch_y})
                sum_time+=time.time()-start_time
                if (i>0 and i % show == 0):
                    # batch_x, batch_y = next_batch(test_x, test_y, update_size)
                    # loss_fine = sess3.run(yhat_3, feed_dict={X_3: batch_x, tst_3: batch_y})
                    print('fine iter  {}    time is  {}        one iter time is {}'.format(i, sum_time,sum_time/show))
                    print('parameters numbers is {}'.format(get_num_params()))
                    print('[fine] Step: %d, loss: %4.5f' % (i, loss_1))
                    sum_time=0.0
            shutil.rmtree("AE1/fashion_model2/")
            saver_.save(sess3,"AE1/fashion_model2/model.ckpt")
            shutil.rmtree('AE1/retrain_logs1/')
            writer = tf.summary.FileWriter('AE1/retrain_logs1/', sess3.graph)
            print('parameters numbers is {}'.format(get_num_params()))
            print('----------------------------------------------Successfully load the model_3_1-----------------success!')
            resources.acquire()
            # save_model(e1_value, e2_value, e3_value, e4_value, e5_value, e6_value, e7_value, e8_value)
            weights_1.encoder_h1 = encoder_h11.eval()
            weights_1.encoder_h2 = encoder_h21.eval()
            weights_1.encoder_b1 = encoder_b11.eval()
            weights_1.encoder_b2 = encoder_b21.eval()
            weights_1.decoder_h1 = decoder_h11.eval()
            weights_1.decoder_h2 = decoder_h21.eval()
            weights_1.decoder_b1 = decoder_b11.eval()
            weights_1.decoder_b2 = decoder_b21.eval()
            # goods锁用于模型文件资源的读写同步
            # print("struct weights:")
            # print("encoder_h11:",weights_1.encoder_b1)
            # print("decoder_h11:", weights_1.decoder_b1)
            # save_model(e1_value,e2_value,e3_value,e4_value,e5_value,e6_value,e7_value,e8_value)
            resources.release()
            goods.sub()
    return sess3,X_3,tst_3,decoder2_3

