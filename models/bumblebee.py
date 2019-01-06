from base import BaseModel

from utils import *

import numpy as np
try:
	import _pickle as cPickle
except:
	import cPickle
import cv2
import tensorflow as tf
import os, time, sys, math, random

class BumbleBee(BaseModel):
	def __init__(self):
		config = {'max_iters': 10, 'debug': True}
		super(BumbleBee, self).__init__(config)

	def init(self):
		self.load_dataset()
		self.build_graph()
		initializer = tf.global_variables_initializer()
		self.load_saved_model()
		self.session.run(initializer)



	def load_saved_model(self):
		return True
		saver = tf.train.Saver()
		saver.restore(self.session, os.path.join(os.path.getcwd(), 'results', 'results.ckpt'))


	def load_dataset(self):
		self.data_info = readDataSetInfo()
		
		sample_image, sample_small_image, sample_rotation = readImages(self.data_info, batch_size=1, limit=2000).next()

		self.image_height = sample_small_image.shape[0]
		self.image_width = sample_small_image.shape[1]
		self.image_depth = sample_small_image.shape[2]

	def predict(self, images):
		test_data = {x: images}
		predicted_value = sess.run(self.y_predicted, feed_dict=test_data)
		# for i in range(len(predicted_value)):
		# 	print str(predicted_value[i]).ljust(15)+ str(actual_value[i]).ljust(15)
		return predicted_value.flatten()

	def learn_from_epoch(self):
		for original_images, batch_X, batch_Y, in readImages(self.data_info, batch_size=10):
			batch_X = preprocessData(batch_X)
			self.train_data = {self.x: batch_X, self.y_actual: batch_Y}
		
			# Training Step
			self.session.run(self.optimizer, feed_dict=self.train_data)
		
			# if counter % 500 == 0:
			#     clear_output(wait=True)
			#     display('Iteration '+str(i)+ ' Batch counter ' + str(counter) + ' Entropy: '+ str(entropy_value))
			# counter += 1
		
		# if not np.isnan(entropy_value):
		# 	entropy.append(entropy_value)
		# 	draw(fig, ax, entropy)
		# 	print('Iteration '+str(i))

	def calculate_metrics(self, epoch, time_taken):
		entropy_value = self.session.run(self.cross_entropy, feed_dict=self.train_data)
		print('The iteration is %d/%d : Entropy :%f, Time taken : %d'%(epoch, self.max_iters, entropy_value, time_taken))

	def build_graph(self):
		filter_size_2 = 2
		filter_size_3 = 3
		filter_size_5 = 5
		filter_count_24 = 24
		filter_count_32 = 32
		filter_count_36 = 36
		filter_count_48 = 48
		filter_count_64 = 64
		filter_count_128 = 128

		fully_conn_layer_1_out_size = 100
		fully_conn_layer_2_out_size = 50
		fully_conn_layer_3_out_size = 10
		fully_conn_layer_4_out_size = 1

		self.x = tf.placeholder(tf.float32, shape=[None, self.image_height, self.image_width, self.image_depth])
		self.y_actual = tf.placeholder(tf.float32, shape=(None, 1))

		conv_layer_1, conv_weights_1 = createConvulationalLayer(input_data=self.x,
						   channel_count=self.image_depth,
						   filter_size=filter_size_5,
						   filter_count=filter_count_24,
						   max_pooling=False)

		conv_layer_2, conv_weights_2 = createConvulationalLayer(input_data=conv_layer_1,
						   channel_count=filter_count_24,
						   filter_size=filter_size_5,
						   filter_count=filter_count_36,
						   max_pooling=True)

		conv_layer_3, conv_weights_3 = createConvulationalLayer(input_data=conv_layer_2,
						   channel_count=filter_count_36,
						   filter_size=filter_size_5,
						   filter_count=filter_count_48,
						   max_pooling=False)

		conv_layer_4, conv_weights_4 = createConvulationalLayer(input_data=conv_layer_3,
						   channel_count=filter_count_48,
						   filter_size=filter_size_3,
						   filter_count=filter_count_64,
						   max_pooling=True)

		conv_layer_5, conv_weights_5 = createConvulationalLayer(input_data=conv_layer_4,
						   channel_count=filter_count_64,
						   filter_size=filter_size_3,
						   filter_count=filter_count_64,
						   max_pooling=False)


		flat_layer, num_features = createFlattenLayer(conv_layer_5)


		fully_con_layer_1 = createFullyConnectedLayer(input_data=flat_layer,
								 input_feature_count=num_features,
								 output_feature_count=fully_conn_layer_1_out_size,
								 apply_relu=True)


		fully_con_layer_2 = createFullyConnectedLayer(input_data=fully_con_layer_1,
								 input_feature_count=fully_conn_layer_1_out_size,
								 output_feature_count=fully_conn_layer_2_out_size,
								 apply_relu=True)


		fully_con_layer_3 = createFullyConnectedLayer(input_data=fully_con_layer_2,
								 input_feature_count=fully_conn_layer_2_out_size,
								 output_feature_count=fully_conn_layer_3_out_size,
								 apply_relu=False)


		fully_con_layer_4 = createFullyConnectedLayer(input_data=fully_con_layer_3,
								 input_feature_count=fully_conn_layer_3_out_size,
								 output_feature_count=fully_conn_layer_4_out_size,
								 apply_relu=False)


		self.y_predicted = fully_con_layer_4 # tf.nn.softmax(fully_con_layer_4)
		self.cross_entropy = tf.reduce_mean(tf.square(tf.subtract(self.y_actual, self.y_predicted)))
		# self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_actual * tf.log(self.y_predicted), reduction_indices=[1]))
		self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)


		is_correct = tf.equal(tf.argmax(self.y_predicted, 1), tf.argmax(self.y_actual, 1))
		self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

