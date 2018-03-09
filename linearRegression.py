import numpy as np
import matplotlib.pyplot as plt
NUM_POINTS = 1000
vectors = []
b1 = 0.3
w1 = 0.1
for i in range(NUM_POINTS):
    x1 = np.random.normal(0.0,0.55)
    y1 = x1 * w1 + b1 + np.random.normal(0.0,0.05)
    vectors.append([x1,y1])
x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]



def plot(x_data,y_data):
    plt.plot(x_data,y_data,'ro',label='original data')
    plt.legend()
    plt.show()


import tensorflow as tf
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros(1))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(8):
        sess.run(train)
        #W,b 逼近于 w1,b1
        #print(step,sess.run(W),sess.run(b),sess.run(loss))

        plt.plot(x_data,y_data,'ro')
        plt.plot(x_data,sess.run(W) * x_data + sess.run(b))
        plt.xlabel('x')
        plt.xlim(-2,2)
        plt.ylim(0.1,0.6)
        plt.ylabel('y')
        plt.legend()
        plt.show()

