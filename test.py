# from mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# print(mnist.train.images)
# print(mnist.train.labels)

import tensorflow as tf
#
# x = tf.placeholder(tf.string)
# y = tf.placeholder(tf.int32)
# z = tf.placeholder(tf.float32)
#
# with tf.Session() as sess:
#     output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
#
#     print(output)
#
# tf.reduce_sum()

# fname = "bgs/{:08d}.jpg".format(1)
#
# print(fname)
#
# a=tf.placeholder("float")
# b=tf.placeholder("float")
# y=tf.multiply(a,b)
#
# t = tf.constant([[[1, 1, 1], [2, 2, 2]],
#                    [[3, 3, 3], [4, 4, 4]],
#                    [[5, 5, 5], [6, 6, 6]]])
# y1=tf.slice(t, [1, 0, 0], [1, 2, 3]);
#
# with tf.Session() as sess:
#     print(sess.run({"y":y1},feed_dict={}))
#
# tf.diag()

w1 = tf.Variable(tf.random_normal([2,3],stddev=1.0))
b1 = tf.Variable(tf.constant(0.0,shape=[3]))
x1 = tf.constant([[0.7, 0.9]])
y1 = tf.nn.relu(tf.matmul(x1,w1) + b1)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y1))



