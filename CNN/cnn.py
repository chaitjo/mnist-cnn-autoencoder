import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.optimizers import Momentum
import tensorflow as tf
import os

tf.flags.DEFINE_string("learning_rate", 0.01)
tf.flags.DEFINE_string("momentum", 0.9)
tf.flags.DEFINE_string("lr_decay", 0.96)
tf.flags.DEFINE_string("batch_size", 50)

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

sgd_momentum = Momentum(learning_rate=learning_rate, momentum=momentum, 
						lr_decay=lr_decay, decay_step=100)
network = regression(network, optimizer=sgd_momentum, learning_rate=learning_rate,
					loss='categorical_crossentropy', metric='accuracy', name='target')

# Training
model_name = 'sgdm_1layer_lr{}_mom{}_dec{}_batch{}'.format(learning_rate, momentum, lr_decay, batch_size)
os.makedirs('/checkpoints/'+model_name)
os.makedirs('/best_checkpoints/'+model_name)

model = tflearn.DNN(network, 
					tensorboard_verbose=0, tensorboard_dir='summaries', 
					checkpoint_path='checkpoints/'+model_name+'/checkpoints', 
					best_checkpoint_path='best_checkpoints/'+model_name+'/best_checkpoints', best_val_accuracy=0.85)
model.fit({'input': X}, {'target': Y}, n_epoch=25,
		validation_set=({'input': testX}, {'target': testY}),
		batch_size=batch_size, shuffle=True,
		snapshot_step=50, snapshot_epoch=True,
		show_metric=True, run_id=model_name)
