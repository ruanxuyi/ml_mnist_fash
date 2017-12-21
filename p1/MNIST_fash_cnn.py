# -*- coding: utf-8 -*-
# @Author: xruan
# @Date:   2017-10-28 17:10:27
# @Last modified by:   Xuyi Ruan
# @Last Modified time: 2017-11-17 14:54:28w
# modifed from: https://www.tensorflow.org/get_started/mnist/pros

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from numpy import concatenate, mean, asarray
from sklearn.decomposition import PCA
import time

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# load the MNIST data by TensorFlow
mnist = input_data.read_data_sets("../MNIST_data/fashion", one_hot=True, validation_size=0)

# start interactive session
# sess = tf.InteractiveSession()

# input/output value (called placeholder in tf) for TensorFlow to run a computation 
# [None] indicates that the first dimension, corresponding to the batch size, can be of any size.
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
print("x:", x.shape)

# (5,5,1,32)
# 5x5 filter size
# 1 filter_in dimension, x_image greyscale, just one layer (should be [same] as X/in image's depth (ex. rgb = 3 layers)
# 32 # of filters
# dimension of b matches dimen of w's output depth dimension
W_conv1 = weight_variable([5, 5, 1, 32]) # requirement from tf.nn.conv2d()
print("W_conv1:", W_conv1.shape)

b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1]) # requirement from tf.nn.conv2d()
print("x_image:", x_image.shape)

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
print("h_conv1:", h_conv1.shape)

h_pool1 = max_pool_2x2(h_conv1)
print("h_pool1:", h_pool1.shape)


# (5,5,32,64)
# 5x5 filter size
# 32 filter_in dimension (should be [same] as X/in image's depth (14, 14, 32)
# 64 # of filters
# dimension of b matches dimen of w's output depth dimension
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

print("h_pool1:", h_pool1.shape)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # tf.conv2d()

print("h_conv2:", h_conv2.shape)
h_pool2 = max_pool_2x2(h_conv2)
print("h_pool2:", h_pool2.shape)

# dimension of b matches dimen of w's output depth dimension
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
print("h_pool2_flat:", h_pool2_flat.shape)

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # tf.matmul()
print("h_fc1:", h_fc1.shape)

keep_prob = tf.placeholder(tf.float32)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print("h_fc1_drop:", h_fc1_drop.shape)


W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])


y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print("y_conv:", y_conv.shape)

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# record time
time_start = time.time()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(100): # defult 20000
    batch = mnist.train.next_batch(50)
    print("batch[0]", batch[0].shape, "batch[1]", batch[1].shape)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) # keep_prob controls dropout rate

  # time used
  time_end=time.time()

  print('Time to train: %0.2f minuites.' % ((time_end-time_start)/60))

  print('Test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

for op in tf.get_default_graph().get_operations(): #打印模型节点信息
    print(op.name,op.values())
