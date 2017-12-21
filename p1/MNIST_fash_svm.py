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
import time
from collections import Counter


LINEAR = '-t 0'
POLYNOMIAL = '-t 1 -c 10'
RBF = '-c 10 -t 2'
RBF_CV = '-c 10 -t 2 -v 10'

def main():
	TRAIN_KERNEL = POLYNOMIAL # choice: LINEAR, POLYNOMIAL, RBF
	MODEL_OUT_NAME = 'MINST_svm_poly_tmp.model'
	PCA_component = 40

	############################## Prepare data ########################################
	image_train, label_train, image_test, label_test = prepare_data()

	############################## Dimension reduction #################################
	# PCA dimension reduction
	reduced_dimension_images = reduce_dim_pca(PCA_component, image_train, image_test)
	image_train = reduced_dimension_images[0]
	image_test = reduced_dimension_images[1]

	# LDA dimension reduction
	# # reduced_dimension_images = reduce_dim_lda(image_train, label_train, image_test, 9)
	# reduced_dimension_images = reduce_dim_lda(image_train, label_train, image_test)
	# image_train = reduced_dimension_images[0]
	# image_test = reduced_dimension_images[1]

	# array to list
	image_train = image_train.tolist()
	image_test = image_test.tolist()
	label_train = label_train.tolist()
	label_test = label_test.tolist()

	# record time
	time_start = time.time() 

	############################### SVM Train ###########################################
	label_predict, accuracy, decision_values = SVM(image_train, label_train, image_test, label_test, TRAIN_KERNEL)
	
	############################### SVM Test ############################################
	print "Accuracy!: ", accuracy

	# time used
	time_end=time.time()
	print('Time to classify: %0.2f minuites.' % ((time_end-time_start)/60))

def prepare_data():
	# load the MNIST data by TensorFlow
	mnist = input_data.read_data_sets("../MNIST_data/fashion", one_hot=False)

	image_train = mnist.train.images
	image_validation = mnist.validation.images
	image_test = mnist.test.images

	label_train = mnist.train.labels
	label_validation = mnist.validation.labels
	label_test = mnist.test.labels

	# merge the training and validation datasets
	image_train = concatenate((image_train, image_validation), axis=0)
	label_train = concatenate((label_train, label_validation), axis=0)

	return [image_train, label_train, image_test, label_test]

def SVM(image_train, label_train, image_test, label_test, TRAIN_KERNEL):
	# linear SVM by Libsvm in Python
	model = svm_train(label_train, image_train, TRAIN_KERNEL) # '-t 0 linear'
	# svm_save_model(MODEL_OUT_NAME, model)

	return svm_predict(label_test, image_test, model)

def reduce_dim_pca(PCA_component, train_data, test_data):
	print "PCA processing..."

	pca = PCA(n_components=PCA_component)
	pca.fit(train_data)
	image_train_pca = pca.transform(train_data)
	image_test_pca = pca.transform(test_data)

	print "PCA done..."
	return [image_train_pca, image_test_pca]

def reduce_dim_lda(train_data, train_label, test_data, n_components=None):
	print "LDA processing..."
	lda = LinearDiscriminantAnalysis(n_components=n_components)
	lda.fit(train_data, train_label)

	image_train_pca = lda.transform(train_data)
	image_test_pca = lda.transform(test_data)

	print "LDA done..."

	return [image_train_pca, image_test_pca]

if __name__ == '__main__':
	main()

