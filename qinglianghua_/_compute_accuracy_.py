#计算微调后的模型准确率
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np

def compute_accuracy_(v_xs,v_ys,sess3,X_3,tst_3,decoder2_3):
    #prediction 变为全剧变量
    # global X_3, tst_3, yhat_3,decoder2_3
    with sess3.as_default():
        with sess3.graph.as_default():
            correct_prediction = tf.equal(tf.argmax(decoder2_3,1),tf.argmax(tst_3,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            # 运行我们的accuracy这一步
            result = sess3.run(accuracy,feed_dict={X_3:v_xs,tst_3:v_ys})
            confusion_matrix1 = confusion_matrix(tf.argmax(sess3.run(tst_3,feed_dict={tst_3:v_ys}),1).eval(), tf.argmax(sess3.run(decoder2_3,feed_dict={X_3:v_xs}),1).eval())
            # print(confusion_matrix1)
            line=np.zeros(10)
            precision=np.zeros(10)
            for i in range(10):
                for j in range(10):
                    line[i] += confusion_matrix1[i][j]
            for i in range(10):
                precision[i] = float(confusion_matrix1[i][i]) / line[i]
            # print('precision:',precision)
            # print('W_fc2:',W_fc2.eval())
            return result,precision
