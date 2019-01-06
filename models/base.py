import os, json
from copy import deepcopy
from datetime import datetime

import numpy as np
try:
	import _pickle as cPickle
except:
	import cPickle
import cv2
import tensorflow as tf
import time, sys, math, random

class BaseModel(object):
	def __init__(self, config={'max_iters': 10}):
		self.config = deepcopy(config)

		if self.config['debug']:
			print('Configs : ', self.config)

		self.max_iters = config['max_iters']

		self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))

		self.init()


	def init(self):
		raise Exception('The init function needs to be overridden by the class')

	def build_graph(self):
		raise Exception('The build_graph function needs to be overridden by the class')

	def learn_from_epoch(self):
		raise Exception('The learn_from_epoch function needs to be overridden by the class')

	def calculate_metrics(self, epoch):
		raise Exception('The calculate_metrics function needs to be overridden by the class')

	def train(self):
		for epoch in range(self.max_iters):
			self.learn_from_epoch()

			self.calculate_metrics(epoch)

	def predict(self):
		raise Exception('The predict function needs to be overridden by the class')

	def save(self):
		results_base_dir = os.path.join(os.getcwd(), 'results')
		if not os.path.exists(results_base_dir):
			os.makedirs(results_base_dir)
		
		current_timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
		results_dir = os.path.join(results_base_dir, current_timestamp)
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)

		saver = tf.train.Saver()
		save_path = saver.save(self.session, os.path.join(results_dir, 'results.ckpt'))

		with open(os.path.join(results_dir, 'config.json'), 'w') as f:
			json.dump(self.config, f)




