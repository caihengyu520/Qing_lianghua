#加载初始模型用于微调
import tensorflow as tf
import time
import shutil
from _next_batch import next_batch
from _get_num_params import get_num_params
from _get_var_list import get_var_list
from _next_batch_ import next_batch_
import random

def load_model_3_conv(sess3,test_x,test_y,test_len,resources,goods,weights_1,good_1,optimizer_way):
    """
        Loading the pre-trained model and parameters.
    """
    with sess3.as_default():
        with sess3.graph.as_default():
            print("------------------------------load_model_3")
            modelpath = r'AE1/fashion_model2/'
            saver = tf.train.import_meta_graph(modelpath + 'model.ckpt.meta')
            saver.restore(sess3, tf.train.latest_checkpoint(modelpath))
            graph1 = tf.get_default_graph()
            X_3 = graph1.get_tensor_by_name("x:0")
            tst_3 = graph1.get_tensor_by_name("y:0")
            yhat_3 = graph1.get_tensor_by_name("cross_entropy:0")
            decoder2_3=graph1.get_tensor_by_name("out_final:0")
            encoder2_3=graph1.get_tensor_by_name("h_fpool2:0")
            encoder_h11 = graph1.get_tensor_by_name("w_conv1:0")
            encoder_b11 = graph1.get_tensor_by_name("b_conv1:0")
            encoder_h21 = graph1.get_tensor_by_name("w_conv2:0")
            encoder_b21 = graph1.get_tensor_by_name("b_conv2:0")
            decoder_h11 = graph1.get_tensor_by_name("w_fc1:0")
            decoder_b11 = graph1.get_tensor_by_name("b_fc1:0")
            decoder_h21 = graph1.get_tensor_by_name("w_out:0")
            decoder_b21 = graph1.get_tensor_by_name("b_out:0")
            weights=tf.Variable(tf.random_normal([7*7*32,10]),name="weights_out")
            bias=tf.Variable(tf.random_normal([10]),name="bias_out")
            result=tf.nn.softmax(tf.add(tf.matmul(encoder2_3,weights),bias))
            targ_list = ['w_fc1', 'b_fc1', 'w_out', 'b_out']
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
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tst_3 * tf.log(tf.clip_by_value(result, 1e-8, 1.0)), reduction_indices=[1]),name='cost')
            train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

            if(optimizer_way=="Gradient_descent"):
                train_step_=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,var_list=trg)
            saver_=tf.train.Saver(max_to_keep=1)
            sess3.run(weights.initializer)
            sess3.run(bias.initializer)
            update_1=0
            update_2=50
            show=20
            update_size=20
            index=0
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
            sum_time = 0.0
            for i in range(update_2):
                # start,index_list,batch_x, batch_y = next_batch_(test_x, test_y, update_size,start,index_list)
                batch_x, batch_y = next_batch(test_x, test_y, update_size)
                # print("batch_x:",batch_x)
                # print("batch_y:",batch_y)
                start_time = time.time()
                _, loss_1 = sess3.run([train_step_, yhat_3], feed_dict={X_3: batch_x, tst_3: batch_y})
                sum_time += time.time() - start_time
                if (i > 0 and i % show == 0):
                    # batch_x, batch_y = next_batch(test_x, test_y, update_size)
                    # loss_fine = sess3.run(yhat_3, feed_dict={X_3: batch_x, tst_3: batch_y})
                    print(
                        'fine iter  {}    time is  {}        one iter time is {}'.format(i, sum_time, sum_time / show))

                    # print('parameters numbers is {}'.format(get_num_params()))
                    print('[fine] Step: %d, loss: %4.5f' % (i, loss_1))
                    sum_time = 0.0
            shutil.rmtree("AE1/fashion_model2/")
            saver_.save(sess3,"AE1/fashion_model2/model.ckpt")
            shutil.rmtree('AE1/retrain_logs1/')
            writer = tf.summary.FileWriter('AE1/retrain_logs1/', sess3.graph)
            print('parameters numbers is {}'.format(get_num_params()))
            print('----------------------------------------------Successfully load the model_3------------------success!')
            resources.acquire()
            # save_model(e1_value,e2_value,e3_value,e4_value,e5_value,e6_value,e7_value,e8_value)
            weights_1.encoder_h1 = encoder_h11.eval()
            weights_1.encoder_h2 = encoder_h21.eval()
            weights_1.encoder_b1 = encoder_b11.eval()
            weights_1.encoder_b2 = encoder_b21.eval()
            weights_1.decoder_h1 = decoder_h11.eval()
            weights_1.decoder_h2 = decoder_h21.eval()
            weights_1.decoder_b1 = decoder_b11.eval()
            weights_1.decoder_b2 = decoder_b21.eval()
            #goods锁用于模型文件资源的读写同步
            goods.sub()
            resources.release()

    return X_3,tst_3,decoder2_3
