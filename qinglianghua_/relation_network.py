import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)
tf.set_random_seed(0)

x1 = np.array([1,1,0,
               1,1,0,
               0,0,0])

x2 = np.array([1,1,1,
               1,0,1,
               1,1,1])

x3 = np.array([0,1,0,
               1,1,1,
               0,1,0])

x4 = np.array([0,0,0,
               0,1,1,
               0,1,1])

x5 = np.array([1,0,0,
               1,0,0,
               1,0,0])

x6 = np.array([0,0,0,
               1,1,1,
               0,0,0])

x7 = np.array([0,0,1,
               0,0,1,
               0,0,1])

x8 = np.array([1,0,1,
               0,1,0,
               1,0,1])

x = np.vstack([x1,x2,x3,x4,x5,x6,x7,x8])



for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(x[i].reshape((3,3)))
plt.show()
print("success")

same_class = []
diff_class = []

for i in range(len(x)):
    for j in range(len(x)):
        if i == j:
            same_class.append(np.hstack([x[i], x[j]]))
        else:
            diff_class.append(np.hstack([x[i], x[j]]))

same_class = np.array(same_class)
diff_class = np.array(diff_class)



data = np.vstack([same_class, diff_class])
label = np.vstack([np.ones((len(same_class),1)),np.zeros((len(diff_class),1))])

x1 = tf.placeholder(tf.float32, [None, 9]) # sample image placeholder
x2 = tf.placeholder(tf.float32, [None, 9]) # query image placeholder
y = tf.placeholder(tf.float32, [None, 1]) # label placeholder (note: label is either 0 or 1, but it is a regression problem)

w0 = tf.Variable(tf.truncated_normal([9,1]))
b0 = tf.Variable(tf.truncated_normal([1]))

h1 = tf.nn.elu(tf.nn.xw_plus_b(x1,w0,b0))
h2 = tf.nn.elu(tf.nn.xw_plus_b(x2,w0,b0))

h = tf.concat([h1,h2],axis=1)



w1 = tf.Variable(tf.truncated_normal([2,3]))
b1 = tf.Variable(tf.truncated_normal([3]))

h3 = tf.nn.elu(tf.nn.xw_plus_b(h,w1,b1))



w2 = tf.Variable(tf.truncated_normal([3,1]))
b2 = tf.Variable(tf.truncated_normal([1]))

o = tf.nn.xw_plus_b(h3,w2,b2)
o_normalised = tf.nn.sigmoid(o)

loss = tf.reduce_mean(tf.squared_difference(o_normalised,y))

opt = tf.train.AdamOptimizer(0.1)
train = opt.minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

h_value = h.eval(feed_dict={x1:data[:,0:9], x2:data[:,9:]})
plt.scatter(h_value[:,0], h_value[:,1], c=label.flatten())
plt.show()




for I in range(1000):
    _, loss_value = sess.run([train, loss],
                             feed_dict={x1:data[:,0:9]+np.random.randn(*np.shape(data[:,0:9]))*0.05,
                                        x2:data[:,9:]+np.random.randn(*np.shape(data[:,9:]))*0.05,
                                        y:label})
    if I % 100 == 0:
        print(loss_value)



h_value = h.eval(feed_dict={x1:data[:,0:9], x2:data[:,9:]})
plt.scatter(h_value[:,0], h_value[:,1], c=label.flatten())
plt.colorbar()
plt.show()








