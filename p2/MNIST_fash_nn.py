# -*- coding: utf-8 -*-
# @Author: xruan
# @Date:   2017-11-26 11:05:46
# @Last modified by:   Xuyi Ruan
# @Last Modified time: 2017-11-26 22:35:03w

from tensorflow.examples.tutorials.mnist import input_data
from numpy import concatenate, mean, asarray
from svmutil import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import time
from collections import Counter

PCA_component = 40
KNN = 6

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

# PCA
print("PCA processing...")
pca = PCA(n_components=PCA_component)
pca.fit(image_train)

image_train_pca = pca.transform(image_train)
image_test_pca = pca.transform(image_test)

print("PCA done...")

# array to list
x_train = image_train_pca.tolist()
x_test = image_test_pca.tolist()
y_train = label_train.tolist()
y_test = label_test.tolist()

# record time
time_start = time.time() 

# linear bayes classifier
neigh = KNeighborsClassifier(n_neighbors=KNN, n_jobs=-1)

# Perform the predictions
neigh.fit(x_train, y_train)
# Perform the predictions
y_predicted = clf.predict(x_test)

# int to float
y_predicted = [int(tmp) for tmp in y_predicted] 

# list to array
y_predicted = asarray(y_predicted)
y_test = asarray(y_test)

# accuracy
accuracy = mean((y_predicted == y_test) * 1)
print('n_component: %d' % n_c)
print('Accuracy: %0.4f.' % accuracy)

# time used
time_end=time.time()
#print('Time to classify: %0.2f minuites.' % ((time_end-time_start)/60))
print('Time to classify: %0.2f seconds.' % ((time_end-time_start)))