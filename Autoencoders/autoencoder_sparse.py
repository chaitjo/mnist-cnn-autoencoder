from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf

# Data loading and preprocessing

X, Y = tflearn.data_utils.image_preloader('../Data/Train', image_shape=(28, 28),
										mode='folder', categorical_labels=True)
testX, testY = tflearn.data_utils.image_preloader('../Data/Test', image_shape=(28, 28),
										mode='folder', categorical_labels=True)


X = np.array(X)
X = X.reshape([-1, 784])

testX = np.array(testX)
testX = testX.reshape([-1, 784])

np.random.shuffle(X)
np.random.shuffle(testX)

X = X[:100]
testX = testX[:20]

learning_rate = 0.001
momentum = 0.95
epochs = 100
display_step = 1
n_input = 784
n_hidden = 100

sparsity = 0.05
sparse_coef = 0


inputs = tf.placeholder(tf.float32, [None, n_input])

w1 = tf.Variable(tf.truncated_normal([n_input, n_hidden], dtype=tf.float32, stddev=0.1, name="w1"))
b1 = tf.Variable(tf.constant(value=0., dtype=tf.float32, shape=[n_hidden]))
h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(inputs, w1), b1))

w2 = tf.Variable(tf.truncated_normal(shape=[n_hidden, n_input], dtype=tf.float32, stddev=0.1, name="w2"))
b2 = tf.Variable(tf.constant(value=0., dtype=tf.float32, shape=[n_input]))
outputs = tf.nn.bias_add(tf.matmul(h, w2), b2)

error = tf.reduce_mean(tf.square(inputs - outputs))

active_rates = tf.reduce_mean(h, 0)
kl_divergence = tf.reduce_sum(sparsity * tf.log(sparsity / active_rates) + (1 - sparsity) * tf.log((1 - sparsity) / (1 - active_rates)))

cost = error + sparse_coef * kl_divergence
#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

#reshapedX = X.reshape([-1, 50, 784])
for i in xrange(epochs):
	for record in X:
		sess.run(optimizer, feed_dict={inputs: [record]})
	if i % display_step == 0:
		loss = sess.run(cost, feed_dict={inputs: X})
		test_loss = sess.run(cost, feed_dict={inputs: testX})
		kl = sess.run(kl_divergence, feed_dict={inputs: X})
		activations = sess.run(tf.reduce_mean(active_rates), feed_dict={inputs: X})
		print("epoch %5d, loss %f, test loss %f, KL divergence %f, activations %f" % (i, loss, test_loss, kl, activations))
	weights = sess.run(w1)


# Visualize learned weights
len_ = int(n_hidden ** 0.5)
plt.figure()
for i in xrange(n_hidden):
	plt.subplot(len_, len_, i + 1)
	plt.imshow(weights[:, i].reshape(28, 28), cmap='gray')
	plt.axis('off')
plt.savefig('weights.png')



# Testing the image reconstruction on new data (test set)
encode_decode = sess.run(outputs, feed_dict={inputs: testX})
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(testX[i], (28, 28)), cmap='gray')
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)), cmap='gray')
f.show()
plt.draw()
plt.savefig('reconstruction.png')
