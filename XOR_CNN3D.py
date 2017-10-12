from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import log
import tensorflow as tf
import numpy as np

my_batch = 2
demor=[[0,1],[1,0]]
testr=[[1,0],[0,1]]
demo=[
      [[[0,1],[1,0]],
       [[1,0],[0,0]]],
      [
       [[1,0],[0,1]],
       [[0,1],[1,0]]]
      ]#1&0
'''! ! ! ! ! ! ! ! ! DON'T FORGET EXTRA PORT WHICH TAKES THE PLACE OF BATCHSIZE ! ! ! ! ! ! ! !'''
test1=[[
       [[0,0],[0,0]],
       [[1,1],[1,1]]
       ]]#0
test2=[[
       [[1,0],[1,0]],
       [[0,0],[0,0]]
       ]]#1

'''DEFINE PARAMENTS INITIAL FUNCTION'''
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1,name=name)
    return tf.Variable(initial)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape,name=name)
    return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2x2(x):
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')

x_image = tf.placeholder(tf.float32, [my_batch, 2,2,2])
x_resh  = tf.reshape(x_image,[my_batch,2,2,2,1])
p_image = tf.placeholder(tf.float32, [1,2,2,2])
p_resh  = tf.reshape(p_image,[-1,2,2,2,1])

W_conv1 = weight_variable([2,2,2,1,32],name="W_conv1")
B_conv1 = bias_variable([32],name="B_conv1")
h_conv1 = tf.nn.relu(conv2d(x_resh,W_conv1) + B_conv1)
p_conv1 = tf.nn.relu(conv2d(p_resh,W_conv1) + B_conv1)

h_pool1 = max_pool_2x2x2(h_conv1)
p_pool1 = max_pool_2x2x2(p_conv1)

W_fc1   = weight_variable([32,64],name="W_fc1")
B_fc1   = bias_variable([64],name="B_fc1")
h_pool2_flat = tf.reshape(h_pool1, [-1, 32])
p_pool2_flat = tf.reshape(p_pool1, [-1, 32])

h_fc1   = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + B_fc1)
p_fc1   = tf.nn.relu(tf.matmul(p_pool2_flat,W_fc1) + B_fc1)

W_fc2   = weight_variable([64,2],name="W_fc2")
B_fc2   = bias_variable([2],name="B_fc2")
y_conv  = tf.nn.softmax(tf.matmul(h_fc1,W_fc2) + B_fc2)
p_conv  = tf.nn.softmax(tf.matmul(p_fc1,W_fc2) + B_fc2)

y_      = tf.placeholder(tf.float32,[my_batch,2])

'''MODEL TRAINING OPS OBJECTS'''
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4,).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(2000):
        train_step.run(feed_dict = {x_image: demo, y_: demor})
    score=p_conv.eval(feed_dict={p_image: test1})
    print (score)

'''HAHAHAHAHAHAHAHAHAHAHAAH SUCCESS PASS THE TEST'''