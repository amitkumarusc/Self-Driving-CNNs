# coding: utf-8

import numpy as np
try:
	import _pickle as cPickle
except:
	import cPickle
import cv2
import tensorflow as tf
import os, time, sys, math, random


# import matplotlib
# from matplotlib import pyplot as plt
# plt.rcParams['figure.figsize'] = (16, 9)
# plt.style.use('ggplot')


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(length):
	initial = tf.constant(0.1, shape=[length], dtype=tf.float32)
	return tf.Variable(initial)


def createConvulationalLayer(input_data, channel_count, filter_size, filter_count, max_pooling=True):
	shape = [filter_size, filter_size, channel_count, filter_count]
	weights = weight_variable(shape=shape)
	bias = bias_variable(length=filter_count)
	layer = tf.nn.conv2d(input=input_data,
						 filter=weights,
						 strides=[1, 1, 1, 1],
						 padding='SAME')
	layer += bias
	if max_pooling:
		layer = tf.nn.max_pool(value=layer,
							   ksize=[1, 2, 2, 1],
							   strides=[1, 2, 2, 1],
							   padding='SAME')
		layer = tf.layers.dropout(layer, rate=0.25)
	layer = tf.nn.relu(layer)
	return layer, weights


def createFlattenLayer(layer):
	layer_shape = layer.get_shape()
	# layer_shape is assumed to be in shape [number_of_images, image_height, image_width, channels]
	# The channels will be the number of fiter_count in the previous layer
	feature_count = layer_shape[1:4].num_elements()
	# layer_shape[1:4].num_elements()
	flat_layer = tf.reshape(layer, [-1, feature_count])
	return flat_layer, feature_count


def createFullyConnectedLayer(input_data, input_feature_count, output_feature_count, apply_relu=False):
	weights = weight_variable(shape=[input_feature_count, output_feature_count])
	bias = bias_variable(length=output_feature_count)
	layer = tf.matmul(input_data, weights) + bias
	if apply_relu:
		layer = tf.nn.relu(layer)
	return layer


def splitFilenameAndAngle(row):
	filename_and_angle = row.strip().split(',')[0].split()
	result = [filename_and_angle[0], (float(filename_and_angle[1]) * math.pi) / 180]
	return result


def readDataSetInfo(dataset_id='01', shuffle=True, start_index=0):
	info_data = ''
	info_data_dict = {}
	dataset_path = os.path.join('data', 'dataset_' + dataset_id)
	info_file_path = os.path.join(dataset_path, 'data.txt')
	print dataset_path
	xs = []
	ys = []
	with open(info_file_path, 'r') as f:
		for line in f.readlines():
			filename_and_angle = line.strip().split(',')[0].split()
			x, y = filename_and_angle[0], (float(filename_and_angle[1]) * math.pi) / 180
			xs.append(x)
			ys.append(y)
	
	together = list(zip(xs, ys))

	if shuffle:
		random.shuffle(together)
	filenames, angles = zip(*together)
	
	info_data_dict['path'] = dataset_path
	info_data_dict['filenames'] = filenames[start_index:]
	info_data_dict['angles'] = angles[start_index:]
	return info_data_dict


def readImages(dataset_info, limit=2000, batch_size=40):
	image_data_list = []
	small_image_data_list = []
	rotation_data_list = []
	for index, filename in enumerate(dataset_info['filenames']):
		if index > limit:
			break
		image_path = os.path.join(dataset_info['path'], 'images', filename)
		image_data = cv2.imread(image_path)
		small_image_data = cv2.resize(image_data, (0,0), fx=0.4, fy=0.4)
		small_image_data_list.append(small_image_data)
		image_data_list.append(image_data)

		rotation_data_list.append(dataset_info['angles'][index])
		if len(image_data_list) == batch_size:
			if batch_size == 1:
				image_data_list, small_image_data_list, rotation_data_list = image_data_list[0], small_image_data_list[0], rotation_data_list[0]
			yield np.array(image_data_list), np.array(small_image_data_list), np.array(rotation_data_list).reshape(-1, 1)
			image_data_list = []
			small_image_data_list = []
			rotation_data_list = []
		
	if image_data_list:
		yield np.array(image_data_list), np.array(small_image_data_list), np.array(rotation_data_list).reshape(-1, 1)

def preprocessData(x):
	mu = np.mean(x, axis=(0, 1, 2), keepdims=1)
	sigma = np.std(x, axis=(0, 1, 2), keepdims=1)
	x = x - mu
	x = x / sigma
	return x

def loadModel():
	sess = tf.Session()
	saver = tf.train.Saver()
	saver.restore(sess, os.path.join("model", "model.ckpt"))
	return sess


def predict(images, actual_value=''):
	test_data = {x: images}
	predicted_value = sess.run(y_predicted, feed_dict=test_data)
	# for i in range(len(predicted_value)):
	# 	print str(predicted_value[i]).ljust(15)+ str(actual_value[i]).ljust(15)
	return predicted_value.flatten()
