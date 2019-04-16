#恢复模型，从已经微调过的模型开始恢复
import tensorflow as tf
import time
import shutil
from _next_batch import next_batch
from _get_var_list import get_var_list
from _fine_update import fine_update
from _get_num_params import get_num_params
# from _save_model import save_model
# from _compute_accuracy_ import compute_accuracy_

#修改过后的tensorflow模型保存
def load_model_3_1(sess3,X_3,tst_3,decoder2_3,encoder2_3,test_x,test_y,test_len,resources,goods,weights_1):
    """
        Loading the pre-trained model and parameters.
    """

    # global X_3, tst_3, yhat_3,decoder2_3,encoder2_3
    with sess3.as_default():
        with sess3.graph.as_default():
            print("------------------------------load_model_3_1")
            modelpath = r'AE1/model2/'
            saver = tf.train.import_meta_graph(modelpath + 'model.ckpt.meta')
            saver.restore(sess3, tf.train.latest_checkpoint(modelpath))
            graph = tf.get_default_graph()
            X_3 = graph.get_tensor_by_name("xs:0")
            tst_3 = graph.get_tensor_by_name("ys:0")
            yhat_3 = graph.get_tensor_by_name("cross_entropy:0")
            cost=graph.get_tensor_by_name("cost:0")
            decoder2_3=graph.get_tensor_by_name("decoder2:0")
            # encoder2_3=graph.get_tensor_by_name("encoder2:0")
            # decoder1_3=graph.get_tensor_by_name("decoder1:0")
            # encoder1_3=graph.get_tensor_by_name("encoder1:0")
            encoder_h1=graph.get_tensor_by_name("encoder_h1:0")
            encoder_b1=graph.get_tensor_by_name("encoder_b1:0")
            encoder_h2=graph.get_tensor_by_name("encoder_h2:0")
            encoder_b2=graph.get_tensor_by_name("encoder_b2:0")
            decoder_h1=graph.get_tensor_by_name("decoder_h1:0")
            decoder_b1=graph.get_tensor_by_name("decoder_b1:0")
            decoder_h2=graph.get_tensor_by_name("decoder_h2:0")
            decoder_b2=graph.get_tensor_by_name("decoder_b2:0")
            # weights_out = graph.get_tensor_by_name("weights_out:0")
            # bias_out = graph.get_tensor_by_name("bias_out:0")
            saver_=tf.train.Saver(max_to_keep=1)
            targ_list=['decoder_h1','decoder_h2','decoder_b1','decoder_b2']
            var_list=list(tf.trainable_variables())
            trg=list(get_var_list(targ_list,var_list))
            # result=tf.nn.softmax(tf.add(tf.matmul(encoder2_3,weights_out),bias_out))
            train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
            train_step_=tf.train.GradientDescentOptimizer(1e-2).minimize(yhat_3,var_list=trg)

            batch_size=50
            update_1=0
            update_2=100
            show=10
            update_size=50
            index=0
            # for i in range(update_1):
            #     index, batch_x, batch_y = fine_update(test_x, test_y, update_size, test_len, index)
            #     _, loss_1 = sess3.run([train_step, cost], feed_dict={X_3: batch_x, tst_3: batch_y})
            #     if (i % show == 0):
            #         index, batch_x, batch_y = fine_update(test_x, test_y, update_size, test_len, index)
            #         start_time = time.time()
            #         _, loss_train = sess3.run([train_step, cost], feed_dict={X_3: batch_x, tst_3: batch_y})
            #         duration = time.time() - start_time
                    # print('train iter  {}    time is  {}'.format(i, duration))
                    # print('parameters numbers is {}'.format(get_num_params()))
                    # print('[Train] Step: %d, loss: %4.5f' % (i, loss_train))
            # print("decoder_h1:",sess3.run(decoder_h1))
            # print("encoder_h1:",sess3.run(encoder_h1))
            for i in range(update_2):
                batch_x, batch_y = next_batch(test_x, test_y, update_size)
                _, loss_1 = sess3.run([train_step_, yhat_3], feed_dict={X_3: batch_x, tst_3: batch_y})
                if (i % show == 0):
                    batch_x, batch_y = next_batch(test_x, test_y, update_size)
                    start_time = time.time()
                    _, loss_fine = sess3.run([train_step_, yhat_3], feed_dict={X_3: batch_x, tst_3: batch_y})
                    duration = time.time() - start_time
                    # print('fine iter  {}    time is  {}'.format(i, duration))
                    # print('parameters numbers is {}'.format(get_num_params()))
                    # print('[fine] Step: %d, loss: %4.5f' % (i, loss_fine))
            # print("decoder_h1:",sess3.run(decoder_h1))
            # print("encoder_h1:",sess3.run(encoder_h1))
            shutil.rmtree("AE1/model2/")
            saver_.save(sess3,"AE1/model2/model.ckpt")
            # result_=sess3.run(result,feed_dict={X_3:test_x})
            e1_value=sess3.run(encoder_h1)
            e2_value = sess3.run(encoder_h2)
            e3_value = sess3.run(encoder_b1)
            e4_value = sess3.run(encoder_b2)
            e5_value = sess3.run(decoder_h1)
            e6_value = sess3.run(decoder_h2)
            e7_value = sess3.run(decoder_b1)
            e8_value = sess3.run(decoder_b2)
            shutil.rmtree('AE1/retrain_logs1/')
            writer = tf.summary.FileWriter('AE1/retrain_logs1/', sess3.graph)
            print('parameters numbers is {}'.format(get_num_params()))
            print('----------------------------------------------Successfully load the model_3_1-----------------success!')
            resources.acquire()
            # save_model(e1_value, e2_value, e3_value, e4_value, e5_value, e6_value, e7_value, e8_value)
            weights_1.encoder_h1 = encoder_h1.eval()
            weights_1.encoder_h2 = encoder_h2.eval()
            weights_1.encoder_b1 = encoder_b1.eval()
            weights_1.encoder_b2 = encoder_b2.eval()
            weights_1.decoder_h1 = decoder_h1.eval()
            weights_1.decoder_h2 = decoder_h2.eval()
            weights_1.decoder_b1 = decoder_b1.eval()
            weights_1.decoder_b2 = decoder_b2.eval()
            print("e8_value:",e8_value)
            print("weights_value:",weights_1.decoder_b2)
            # goods锁用于模型文件资源的读写同步
            goods.sub()
            resources.release()


    return X_3,tst_3,decoder2_3,e1_value,e2_value,e3_value,e4_value,e5_value,e6_value,e7_value,e8_value

