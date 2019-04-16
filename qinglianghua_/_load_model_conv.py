#加载原始模型
import tensorflow as tf
from _get_num_params import get_num_params
from _compute_accuracy import compute_accuracy
def load_model_conv(sess1,test_x,test_y,weights_1,flag,self_test_x,self_test_y):
    """
        Loading the pre-trained model and parameters.
    """
    # global X_1, tst_1, yhat_1
    tf.reset_default_graph()
    with sess1.as_default():
        with sess1.graph.as_default():
            print("--------------------------------------validation_model--------------------------")
            modelpath = r'AE1/fashion_model/'
            saver = tf.train.import_meta_graph(modelpath + 'model.ckpt.meta')
            saver.restore(sess1, tf.train.latest_checkpoint(modelpath))
            graph1 = tf.get_default_graph()
            X_1 = graph1.get_tensor_by_name("x:0")
            tst_1 = graph1.get_tensor_by_name("y:0")
            # yhat_1 = graph1.get_tensor_by_name("cross_entropy:0")
            decoder2_1 = graph1.get_tensor_by_name("out_final:0")
            encoder_h11 = graph1.get_tensor_by_name("w_conv1:0")
            encoder_b11 = graph1.get_tensor_by_name("b_conv1:0")
            encoder_h21 = graph1.get_tensor_by_name("w_conv2:0")
            encoder_b21 = graph1.get_tensor_by_name("b_conv2:0")
            decoder_h11 = graph1.get_tensor_by_name("w_fc1:0")
            decoder_b11 = graph1.get_tensor_by_name("b_fc1:0")
            decoder_h21 = graph1.get_tensor_by_name("w_out:0")
            decoder_b21 = graph1.get_tensor_by_name("b_out:0")
            if flag==0:
                pass
            else:
                e1 = tf.assign(encoder_h11, weights_1.encoder_h1)
                e2 = tf.assign(encoder_h21, weights_1.encoder_h2)
                e3 = tf.assign(encoder_b11, weights_1.encoder_b1)
                e4 = tf.assign(encoder_b21, weights_1.encoder_b2)
                e5 = tf.assign(decoder_h11, weights_1.decoder_h1)
                e6 = tf.assign(decoder_h21, weights_1.decoder_h2)
                e7 = tf.assign(decoder_b11, weights_1.decoder_b1)
                e8 = tf.assign(decoder_b21, weights_1.decoder_b2)
                # sess1.run(tf.global_variables_initializer())
                sess1.run([e1, e2, e3, e4, e5, e6, e7, e8])
            # # result_=sess1.run(decoder2_1,feed_dict={X_1:test_x})
            # # print("decoder2_1:", result_)
            print('Successfully load the model_1!')
            print("model parameters is:",get_num_params())
            print("-----------------------------------validation_model--------------------success")

            # print("zuizhongjieguo:",sess1.run(decoder2_1,feed_dict={X_1:self_test_x}))
            # print("zuizhongjieguo:", sess1.run(tst_1, feed_dict={tst_1: self_test_y}))
            # print("encoder_h11:",sess1.run(encoder_b11))
            # print("decoder_h11:", sess1.run(decoder_b11))
            return decoder2_1,X_1,tst_1