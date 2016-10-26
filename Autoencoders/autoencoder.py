from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf

# Data loading and preprocessing
#import tflearn.datasets.mnist as mnist
#X, Y, testX, testY = mnist.load_data(one_hot=True)


X, Y = tflearn.data_utils.image_preloader('../Data/Train', image_shape=(28, 28),
										mode='folder', categorical_labels=True)
testX, testY = tflearn.data_utils.image_preloader('../Data/Test', image_shape=(28, 28),
										mode='folder', categorical_labels=True)

class ReshapedImagePreloader(object):
	def __init__(self, preloader):
		self.preloader = preloader

	def __getitem__(self, id):
		imgs = self.preloader[id]
		imgs = np.array(imgs)
		imgs = imgs.reshape([-1, 784])
		return imgs

	def __len__(self):
		return len(self.preloader)

X = ReshapedImagePreloader(X)
testX = ReshapedImagePreloader(testX)


# Building the encoder
encoder = tflearn.input_data(shape=[None, 784])
hidden = tflearn.fully_connected(encoder, 100, activation='sigmoid', name='hidden')

# Building the decoder
decoder = tflearn.fully_connected(hidden, 784)

sparsity_param = 0.05
sparsity_coeff = 0.5

def kl_divergence(p, p_j):
	return p * tf.log(p) - p * tf.log(p_j) + (1 - p) * tf.log(1 - p) - (1 - p) * tf.log(1 - p_j)

def mean_square_with_sparsity_constraint(y_pred, y_true):
	#u = tf.matmul(hidden.W, X) + hidden.b
	#activations = tf.sigmoid(u)
	#penalty = sparsity_coeff * kl_divergence(sparsity_param, activations)
	penalty = 0
	return tf.reduce_mean(tf.square(y_pred - y_true)) + penalty

# Regression, with mean square error
net = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001,
                         loss=mean_square_with_sparsity_constraint, metric=None)

# Training the auto encoder
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, X, n_epoch=10, validation_set=(testX, testX),
          run_id="auto_encoder", batch_size=256)

# Encoding X[0] for test
print("\nTest encoding of X[0]:")
# New model, re-using the same session, for weights sharing
encoding_model = tflearn.DNN(encoder, session=model.session)
print(encoding_model.predict([X[0][0]]))

# Testing the image reconstruction on new data (test set)
print("\nVisualizing results after being encoded and decoded:")
testX = tflearn.data_utils.shuffle(testX)[0]
testX = [x[0] for x in testX]
# Applying encode and decode over test set
encode_decode = model.predict(testX)
# Compare original images with their reconstructions
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
    a[0][i].imshow(np.reshape(testX[i], (28, 28)))
    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
f.show()
plt.draw()
plt.waitforbuttonpress()
