#加载初始模型用于微调
import tensorflow as tf
import time
import shutil
from _next_batch import next_batch
from _get_num_params import get_num_params
from _get_var_list import get_var_list
from _fine_update import fine_update
# from _save_model import save_model

def load_model_3(sess3,X_3,tst_3,decoder2_3,encoder2_3,test_x,test_y,test_len,resources,goods,weights_1):
    """
        Loading the pre-trained model and parameters.
    """
    # global X_3, tst_3, yhat_3,decoder2_3,encoder2_3
    with sess3.as_default():
        with sess3.graph.as_default():
            print("------------------------------load_model_3")
            modelpath = r'AE1/model2/'
            saver = tf.train.import_meta_graph(modelpath + 'model.ckpt.meta')
            saver.restore(sess3, tf.train.latest_checkpoint(modelpath))
            graph = tf.get_default_graph()
            X_3 = graph.get_tensor_by_name("xs:0")
            tst_3 = graph.get_tensor_by_name("ys:0")
            yhat_3 = graph.get_tensor_by_name("cross_entropy:0")
            decoder2_3=graph.get_tensor_by_name("decoder2:0")
            encoder2_3=graph.get_tensor_by_name("encoder2:0")
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
            weights=tf.Variable(tf.random_normal([128,10]),name="weights_out")
            bias=tf.Variable(tf.random_normal([10]),name="bias_out")
            result=tf.nn.softmax(tf.add(tf.matmul(encoder2_3,weights),bias))
            targ_list=['decoder_h1','decoder_h2','decoder_b1','decoder_b2']
            var_list=list(tf.trainable_variables())
            trg=list(get_var_list(targ_list,var_list))
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(tst_3 * tf.log(tf.clip_by_value(result, 1e-8, 1.0)), reduction_indices=[1]),name='cost')
            train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
            train_step_=tf.train.GradientDescentOptimizer(1e-5).minimize(yhat_3,var_list=trg)
            saver_=tf.train.Saver(max_to_keep=1)
            sess3.run(weights.initializer)
            sess3.run(bias.initializer)
            # result_=sess3.run(result,feed_dict={X_3:test_x})
            # batch_size=100
            update_1=0
            update_2=100
            show=50
            update_size=10
            index=0
            for i in range(update_1):
                index, batch_x, batch_y = fine_update(test_x, test_y, update_size, test_len, index)
                _,loss_1=sess3.run([train_step,cross_entropy],feed_dict={X_3:batch_x,tst_3:batch_y})
                if(i%show==0):
                    index, batch_x, batch_y = fine_update(test_x, test_y, update_size, test_len, index)
                    start_time=time.time()
                    _, loss_train = sess3.run([train_step, cross_entropy], feed_dict={X_3: batch_x, tst_3: batch_y})
                    duration=time.time()-start_time
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
            e1_value=encoder_h1.eval()
            e2_value = encoder_h2.eval()
            e3_value = encoder_b1.eval()
            e4_value = encoder_b2.eval()
            e5_value = decoder_h1.eval()
            e6_value = decoder_h2.eval()
            e7_value = decoder_b1.eval()
            e8_value = decoder_b2.eval()
            shutil.rmtree('AE1/retrain_logs1/')
            writer = tf.summary.FileWriter('AE1/retrain_logs1/', sess3.graph)
            print('parameters numbers is {}'.format(get_num_params()))
            print('----------------------------------------------Successfully load the model_3------------------success!')
            resources.acquire()
            # save_model(e1_value,e2_value,e3_value,e4_value,e5_value,e6_value,e7_value,e8_value)
            weights_1.encoder_h1 = encoder_h1.eval()
            weights_1.encoder_h2 = encoder_h2.eval()
            weights_1.encoder_b1 = encoder_b1.eval()
            weights_1.encoder_b2 = encoder_b2.eval()
            weights_1.decoder_h1 = decoder_h1.eval()
            weights_1.decoder_h2 = decoder_h2.eval()
            weights_1.decoder_b1 = decoder_b1.eval()
            weights_1.decoder_b2 = decoder_b2.eval()
            #goods锁用于模型文件资源的读写同步
            goods.sub()
            resources.release()

    return X_3,tst_3,decoder2_3,e1_value,e2_value,e3_value,e4_value,e5_value,e6_value,e7_value,e8_value
