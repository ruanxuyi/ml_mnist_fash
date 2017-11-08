# -*- coding: utf-8 -*-
# @Author: xruan
# @Date:   2017-10-26 11:05:46
# @Last modified by:   xruan
# @Last Modified time: 2017-11-03 12:40:28w
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


TRAIN_KERNEL = RBF
MODEL_OUT_NAME = 'MINST_svm_poly.model'
PCA_component = 40


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

image_train = pca.transform(image_train)
image_test = pca.transform(image_test)

print("PCA done...")

# array to list
image_train = image_train.tolist()
image_test = image_test.tolist()
label_train = label_train.tolist()
label_test = label_test.tolist()

print("labels: ", Counter(label_test))

# record time
time_start = time.time() 

# linear SVM by Libsvm in Python

model = svm_train(label_train, image_train, TRAIN_KERNEL) # '-t 0 linear'
svm_save_model(MODEL_OUT_NAME, model)

label_predict, accuracy, decision_values = svm_predict(label_test, image_test, model)

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