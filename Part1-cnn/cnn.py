import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf

X, Y = tflearn.data_utils.image_preloader('../Data/Train', image_shape=(28, 28),
	mode='folder', categorical_labels=True)
testX, testY = tflearn.data_utils.image_preloader('../Data/Test', image_shape=(28, 28),
	mode='folder', categorical_labels=True)

class ReshapedImagePreloader(object):
	def __init__(self, preloader):
		self.preloader = preloader

	def __getitem__(self, id):
		img = self.preloader[id]
		img = np.array(img)
		img = img.reshape([-1, 28, 28, 1])
		return img

	def __len__(self):
		return len(self.preloader)

X = ReshapedImagePreloader(X)
testX = ReshapedImagePreloader(testX)

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 20, 9,
				strides=1, padding='same', activation='relu',
				bias=True, weights_init='truncated_normal',
				regularizer='L2', weight_decay=0.001)
network = avg_pool_2d(network, 2)
network = fully_connected(network, 10, activation='softmax',
						bias=True, weights_init='truncated_normal', bias_init='zeros',
						regularizer='L2', weight_decay=0.001)
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=25,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')
