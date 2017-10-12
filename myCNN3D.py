from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import log
import tensorflow as tf
import numpy as np
import random
import Grid

'''class Dataset:
    def __init__(self,data,label,batchsize):
        self.dataset=data
        self.labelset=label
        self.tag=0
        self.len=len(data)
        self.batchsize=batchsize
    def nextbatch(self):
        if self.tag+self.batchsize>=self.len:
            self.tag=0
        else:
            self.tag+=self.batchsize
        return self.dataset[self.tag:self.tag+self.batchsize], self.labelset[self.tag:self.tag+self.batchsize]
'''
class Dataset:
    def __init__(self,batch,path):
        self.batch=batch
        self.path=path
        self.count=0
    def nextbatch(self,batchsize):
        data,label,again=Grid.makeBatch(self.path,self.count,batchsize)
        if again!=1:
            self.count=1
        return data,label

def makeRandomTensor():#shape[depth,height,width]
    cube=[]
    for i in range(48*48*48*32):
        cube.append(random.randint(0,9))
    cube=np.array(cube)
    cube=cube.reshape([48,48,48,32])
    return cube

steps = 2000
my_batch = 2
kernal1 = 5
channel1= 64
kernal2 = 5
channel2= 128
kernal3 = 5
channel3= 32
learnning_rate = 1e-4

#demo, demolabel = Grid.makeBatch("/home/wyd/sample-list.txt")
#test, testlabel = Grid.makeBatch("/test_data_path")
dataset=Dataset(my_batch,"/home/wyd/sample-list.txt")
#dataset2=Dataset(test,testlabel,my_batch)

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
'''POOL LAYER1'''
x_image = tf.placeholder(tf.float32, [my_batch, 48,48,48,32])
x_resh  = tf.reshape(x_image,[my_batch,48,48,48,32])#N_a_a_a_C
unit1_pool  = max_pool_2x2x2(x_resh)#24*24*24*32
'''CONV3D LAYER2'''
unit2_conv_W    = weight_variable([kernal1,kernal1,kernal1,32,channel1],name="unit2_conv_W")
unit2_conv_B    = bias_variable([channel1],name="unit2_conv_B")
unit2_conv_H    = tf.nn.relu(conv2d(unit1_pool,unit2_conv_W) + unit2_conv_B)
'''POOL LAYER3'''
unit3_pool  = max_pool_2x2x2(unit2_conv_H)#12*12*12*64
'''CONV3D LAYER4'''
unit4_conv_W    = weight_variable([kernal2,kernal2,kernal2,channel1,channel2],name="unit4_conv_W")
unit4_conv_B    = bias_variable([channel2],name="unit4_conv_B")
unit4_conv_H    = tf.nn.relu(conv2d(unit3_pool,unit4_conv_W) + unit4_conv_B)
'''POOL LAYER5'''
unit5_pool  = max_pool_2x2x2(unit4_conv_H)#6*6*6*128
'''CONV3D LAYER6'''
unit6_conv_W    = weight_variable([kernal3,kernal3,kernal3,channel2,channel3],name="unit6_conv_W")
unit6_conv_B    = bias_variable([channel3],name="unit6_conv_B")
unit6_conv_H    = tf.nn.relu(conv2d(unit5_pool,unit6_conv_W) + unit6_conv_B)
'''FULL LAYER7'''
unit7_full_flat = tf.reshape(unit6_conv_H,shape=[-1,6*6*6*channel3])
unit7_full_W    = weight_variable([6*6*6*channel3,2],name="unit7_full_W")
unit7_full_B    = bias_variable([2],name="unit7_full_B")
y_conv          = tf.nn.softmax(tf.matmul(unit7_full_flat,unit7_full_W)+unit7_full_B)
y_      = tf.placeholder(tf.float32,[my_batch,2])

'''MODEL TRAINING OPS OBJECTS'''
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(learnning_rate,).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))

'''ADD an op to initialize all the variables'''
init_op = tf.initialize_all_variables()

'''ADD ops to save and restore all the variables.'''
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)

    for i in range(steps):
        data_batch, label_batch = dataset.nextbatch(my_batch)
        train_step.run(feed_dict = {x_image: data_batch, y_: label_batch})

    saver.save(sess, "tmpCNN3D/model.ckpt")
