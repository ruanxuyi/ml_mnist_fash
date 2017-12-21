# -*- coding: utf-8 -*-
# @Author: xruan
# @Date:   2017-10-26 11:05:46
# @Last modified by:   Xuyi Ruan
# @Last Modified time: 2017-11-26 21:40:03w

from tensorflow.examples.tutorials.mnist import input_data
from numpy import concatenate, mean, asarray
from svmutil import *
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
from collections import Counter

def main():
	PCA_component = 50

	############################## Prepare data ########################################
	image_train, label_train, image_test, label_test = prepare_data()

	############################## Dimension reduction #################################
	# PCA dimension reduction
	reduced_dimension_images = reduce_dim_pca(PCA_component, image_train, image_test)
	image_train = reduced_dimension_images[0]
	image_test = reduced_dimension_images[1]

	# LDA dimension reduction
	# reduced_dimension_images = reduce_dim_lda(image_train, label_train, image_test, 9)
	# reduced_dimension_images = reduce_dim_lda(image_train, label_train, image_test)
	# image_train = reduced_dimension_images[0]
	# image_test = reduced_dimension_images[1]

	# array to list
	x_train = image_train.tolist()
	x_test = image_test.tolist()
	y_train = label_train.tolist()
	y_test = label_test.tolist()

	# record time
	time_start = time.time() 

	################################ Bayes Train #########################################
	y_predicted = bayes_train(x_train, y_train, x_test)

	################################ Test Accuracy #######################################
	accuracy_test(y_predicted, y_test)
	
	# time used
	time_end=time.time()
	print('Time to classify: %0.2f seconds.' % ((time_end-time_start)))

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

def bayes_train(x_train, y_train, x_test):
	# linear bayes classifier
	clf = GaussianNB()
	#clf = BernoulliNB()
	#clf = MultinomialNB()

	# Perform the predictions
	clf.fit(x_train, y_train)

	# Perform the predictions
	y_predicted = clf.predict(x_test)
	return y_predicted

def accuracy_test(y_predicted, y_test):
	# int to float
	y_predicted = [int(tmp) for tmp in y_predicted] 

	# list to array
	y_predicted = asarray(y_predicted)
	y_test = asarray(y_test)

	# accuracy
	accuracy = mean((y_predicted == y_test) * 1)
	print('Accuracy: %0.4f.' % accuracy)

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
