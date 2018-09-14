import numpy as np
try:
	import _pickle as cPickle
except:
	import cPickle
import cv2
import tensorflow as tf
import os, time, sys, math, random
import threading, Queue

from test import *

def rotateImage(image, angle):
	image_center = tuple(np.array(image.shape[1::-1]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return result


def predictAngles(q):
	data_info = readDataSetInfo(dataset_id='01', shuffle=False)
	counter = 1
	for original_images, small_images, rotation in readImages(data_info, batch_size=30):
		images = preprocessData(small_images)
		values = predict(images)
		print "Processed : ", 30*counter
		counter += 1
		q.put([original_images, values])


if __name__ == '__main__':
	wheel = cv2.imread('data/wheel.png')
	wheel = cv2.resize(wheel, (0,0), fx=0.5, fy=0.5)
	wheel_shape = wheel.shape
	current_angle = 0
	
	cv2.namedWindow("frame")
	cv2.namedWindow("Win")

	q = Queue.Queue()
	thread = threading.Thread(target=predictAngles, args=(q,))
	thread.start()
	time.sleep(4)

	counter = 1
	while not q.empty():
		images, values = q.get()

		for index, image in enumerate(images):
			cv2.imshow('frame',image)
			print values[index]
			new_angle = (values[index]* 180) / math.pi
			current_angle += 0.2 * pow(abs((new_angle - current_angle)), 2.0 / 3.0) * (new_angle - current_angle) / abs(new_angle - current_angle)
			rotated_mat = cv2.getRotationMatrix2D((wheel_shape[1]/2,wheel_shape[0]/2), -current_angle, 1)
			rotated_wheel = cv2.warpAffine(wheel, rotated_mat, (wheel_shape[1], wheel_shape[0]))
			cv2.imshow("Win", rotated_wheel)
			if cv2.waitKey(60) & 0xFF == ord('q'):
				sys.exit(0)
		print "Read : ", counter * 30
		counter += 1
