# -*- coding: utf-8 -*-
# @Author: xruan
# @Date:   2017-10-28 16:15:19
# @Last modified by:   xruan
# @Last Modified time: 2017-10-28 22:08:53w

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# load the MNIST data by TensorFlow
mnist = input_data.read_data_sets("MNIST_data/fashion", one_hot=True)

# start interactive session
sess = tf.InteractiveSession()

# input/output value (called placeholder in tf) for TensorFlow to run a computation 
# [None] indicates that the first dimension, corresponding to the batch size, can be of any size.
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# define variables (any model parameters in ml)
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# TensorFlow front end using python
# However, its backend relies on a highly efficient C++ backend to do its computation. 
# The connection to this backend is called a session. 
# The common usage for TensorFlow programs is to first create a graph and then launch it in a session.

# initialized variables to be used within the session
sess.run(tf.global_variables_initializer())

######## start of computation graph ###########

# regression model
y = tf.matmul(x,W) + b
# cost function: softmax_cross_entropy_logits ???
cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# train model
# The line of code below added following operations to the computation graph:
# 1. compute gradients, 
# 2. compute parameter update steps, 
# 3. and apply update steps to the parameters.

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

######## end of computation graph ###########


for _ in range(2000):
  batch = mnist.train.next_batch(100) # random select 100 samples for training
  train_step.run(feed_dict={x: batch[0], y_: batch[1]}) #feed_dict updates placeholder x and y_ with random samples

# Evaluate the Model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
