import numpy as np
num_punto = 2000
conjunto_puntos = []
for i in range(num_punto):
    if np.random.random() > 0.5:
        conjunto_puntos.append([np.random.normal(0.0,0.9),np.random.normal(0.0,0.9)])
    else:
        conjunto_puntos.append([np.random.normal(3.0,0.5),np.random.normal(1.0,0.5)])

# print(conjuto_puntos)


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#
# df = pd.DataFrame({"x":[v[0] for v in conjuto_puntos],"y":[v[1] for v in conjuto_puntos]})
# sns.lmplot("x","y",data=df,fit_reg=False,size=6)
# plt.show()

import tensorflow as tf

vectors = tf.constant(conjunto_puntos)
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))
#扩展维度
# expanded_vectors = tf.expand_dims(vectors,1)
# expanded_centroides = tf.expand_dims(centroides,0)
expanded_vectors = tf.expand_dims(vectors,0)
expanded_centroides = tf.expand_dims(centroides,1)
diff = tf.subtract(expanded_vectors,expanded_centroides)
square = tf.square(diff)
reduce_sum = tf.reduce_sum(square,2)
assignments = tf.argmin(reduce_sum,0)
means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in range(k)],0)
update_centroides = tf.assign(centroides,means)


init_op = tf.global_variables_initializer();
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(square))
    # print("-------------")
    # print(sess.run(reduce_sum))
    # print("-------------")
    # print(sess.run(update_centroides))
    for step in range(100):
        _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])

data = {"x": [], "y": [], "cluster": []}
for i in range(len(assignment_values)):
  data["x"].append(conjunto_puntos[i][0])
  data["y"].append(conjunto_puntos[i][1])
  data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
# plt.show()
