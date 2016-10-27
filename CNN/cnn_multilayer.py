import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.optimizers import Momentum
import tensorflow as tf
import os

tf.flags.DEFINE_float("learning_rate", 0.02, "Learning Rate")
tf.flags.DEFINE_float("momentum", 0.99, "Momentum Parameter")
tf.flags.DEFINE_float("lr_decay", 0.95, "LR Decay Parameter")
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size")
tf.flags.DEFINE_integer("filters_hidden1", 30, "Number of filters in hidden layer 1")
tf.flags.DEFINE_integer("filters_hidden2", 50, "Number of filters in hidden layer 2")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

learning_rate = FLAGS.learning_rate
momentum = FLAGS.momentum
lr_decay = FLAGS.lr_decay
batch_size = FLAGS.batch_size
filters_hidden1 = FLAGS.filters_hidden1
filters_hidden2 = FLAGS.filters_hidden2

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
input = input_data(shape=[None, 28, 28, 1], name='input')

convolution1 = conv_2d(input, filters_hidden1, 5,
				strides=1, padding='same', activation='relu',
				bias=True, weights_init='truncated_normal',
				regularizer='L2', weight_decay=0.001)
pooling1 = avg_pool_2d(convolution1, 2)

convolution2 = conv_2d(pooling1, filters_hidden2, 5,
				strides=1, padding='same', activation='relu',
				bias=True, weights_init='truncated_normal',
				regularizer='L2', weight_decay=0.001)
pooling2 = avg_pool_2d(convolution2, 2)

fully_connected = fully_connected(pooling2, 10, activation='softmax',
						bias=True, weights_init='truncated_normal', bias_init='zeros',
						regularizer='L2', weight_decay=0.001)

sgd_momentum = Momentum(learning_rate=learning_rate, momentum=momentum, 
						lr_decay=lr_decay, decay_step=100)
network = regression(fully_connected, optimizer=sgd_momentum, learning_rate=learning_rate,
					loss='categorical_crossentropy', metric='accuracy', name='target')

# Training
model_name = 'sgdm_2layer_{}-{}_lr{}_mom{}_dec{}_batch{}'.format(filters_hidden1, filters_hidden2, learning_rate, momentum, lr_decay, batch_size)
os.makedirs('checkpoints/'+model_name)
os.makedirs('best_checkpoints/'+model_name)

model = tflearn.DNN(network, 
					tensorboard_verbose=0, tensorboard_dir='summaries', 
					checkpoint_path='checkpoints/'+model_name+'/checkpoints', 
					best_checkpoint_path='best_checkpoints/'+model_name+'/best_checkpoints', best_val_accuracy=0.95)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
		validation_set=({'input': testX}, {'target': testY}),
		batch_size=batch_size, shuffle=True,
		snapshot_step=50, snapshot_epoch=True,
		show_metric=True, run_id=model_name)
