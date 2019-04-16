#加载原始模型
import tensorflow as tf
from _get_num_params import get_num_params
from _compute_accuracy import compute_accuracy

def load_model(sess1,test_x,test_y,weights_1,flag):
    """
        Loading the pre-trained model and parameters.
    """
    # 恢复原始模型，通过这个方法可以修改需要恢复的模型，而test_x,test_y数据可以用于原始模型的推断，以判断微调前后准确率的变化
    tf.reset_default_graph()
    with sess1.as_default():
        with sess1.graph.as_default():
            print("--------------------------------------validation_model--------------------------")
            modelpath = r'AE1/model/'
            saver = tf.train.import_meta_graph(modelpath + 'model.ckpt.meta')
            saver.restore(sess1, tf.train.latest_checkpoint(modelpath))
            graph1 = tf.get_default_graph()
            X_1 = graph1.get_tensor_by_name("xs:0")
            tst_1 = graph1.get_tensor_by_name("ys:0")
            # yhat_1 = graph1.get_tensor_by_name("cross_entropy:0")
            decoder2_1 = graph1.get_tensor_by_name("decoder2:0")
            encoder_h11 = graph1.get_tensor_by_name("encoder_h1:0")
            encoder_b11 = graph1.get_tensor_by_name("encoder_b1:0")
            encoder_h21 = graph1.get_tensor_by_name("encoder_h2:0")
            encoder_b21 = graph1.get_tensor_by_name("encoder_b2:0")
            decoder_h11 = graph1.get_tensor_by_name("decoder_h1:0")
            decoder_b11 = graph1.get_tensor_by_name("decoder_b1:0")
            decoder_h21 = graph1.get_tensor_by_name("decoder_h2:0")
            decoder_b21 = graph1.get_tensor_by_name("decoder_b2:0")
            #当采用权重通过结构体恢复模式时需要通过该判断前来进行
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
                sess1.run(tf.global_variables_initializer())
                sess1.run([e1, e2, e3, e4, e5, e6, e7, e8])
            # result_=sess1.run(decoder2_1,feed_dict={X_1:test_x})
            # print("decoder2_1:", result_)
            print('Successfully load the model_1!')
            print("model parameters is:",get_num_params())
            print("-----------------------------------validation_model--------------------success")

            return decoder2_1,X_1,tst_1