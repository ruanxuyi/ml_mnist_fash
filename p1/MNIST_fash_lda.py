# -*- coding: utf-8 -*-
# @Author: xruan
# @Date:   2017-10-26 11:05:46
# @Last modified by:   Xuyi Ruan
# @Last Modified time: 2017-11-18 14:19:26w
# modified from: https://github.com/yuzhounh/MNIST-classification-example-3/blob/master/classify_MNIST.py

from tensorflow.examples.tutorials.mnist import input_data
from numpy import concatenate, mean, asarray
from svmutil import *
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
from collections import Counter

# load the MNIST data by TensorFlow
mnist = input_data.read_data_sets("MNIST_data/fashion", one_hot=False)

image_train = mnist.train.images
image_validation = mnist.validation.images
image_test = mnist.test.images

label_train = mnist.train.labels
label_validation = mnist.validation.labels
label_test = mnist.test.labels

# merge the training and validation datasets
image_train = concatenate((image_train, image_validation), axis=0)
label_train = concatenate((label_train, label_validation), axis=0)

# array to list
image_train = image_train.tolist()
image_test = image_test.tolist()
label_train = label_train.tolist()
label_test = label_test.tolist()

# record time

# linear SVM by Libsvm in Python
for i in range(0, 2):
	lda = LinearDiscriminantAnalysis(n_components=i)
	print("lda started, n_comp=", i)
	
	time_start = time.time() 
	lda = lda.fit(image_train, label_train)
	label_predict = lda.predict(image_test)

	# int to float
	label_predict = [int(tmp) for tmp in label_predict] 

	# list to array
	label_predict = asarray(label_predict)
	label_test = asarray(label_test)

	# accuracy
	accuracy = mean((label_predict == label_test) * 1)
	print('Accuracy: %0.4f.' % accuracy)

	# time used
	time_end=time.time()
	print('Time to classify: %0.2f minuites.' % ((time_end-time_start)/60))
