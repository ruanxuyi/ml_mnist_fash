# -*- coding: utf-8 -*-
# @Author: xruan
# @Date:   2017-11-26 11:05:46
# @Last modified by:   Xuyi Ruan
# @Last Modified time: 2017-11-26 23:22:22w

from tensorflow.examples.tutorials.mnist import input_data
from numpy import concatenate, mean, asarray
import matplotlib.pyplot as plt
import numpy as np

# load the MNIST data by TensorFlow
mnist = input_data.read_data_sets("../MNIST_data/fashion", one_hot=False)

x_train = mnist.train.images
y_train = mnist.train.labels

x_test = mnist.test.images
y_test = mnist.test.labels

def plot_mnist(data, classes):
    for i in range(10):
        idxs = (classes == i)
        
        # get 10 images for class i
        images = data[idxs][0:10]
            
        for j in range(5):   
            plt.subplot(5, 10, i + j*10 + 1)
            plt.imshow(images[j].reshape(28, 28), cmap='gray')
            # print a title only once for each class
            if j == 0:
                plt.title(i)
            plt.axis('off')
    plt.show()

classes = np.argmax(y_train, 1)
plot_mnist(x_train, classes)