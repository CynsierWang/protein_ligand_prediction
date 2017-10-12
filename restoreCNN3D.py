from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import Grid

my_batch = 20

class Dataset:
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

test, testlabel = Grid.makeBatch("/test_data_path")
dataset2 = Dataset(test, testlabel, my_batch)

'''DEFINE PARAMENTS INITIAL FUNCTION'''
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='SAME')

'''POOL LAYER1'''
x_image = tf.placeholder(tf.float32, [my_batch, 48, 48, 48, 32])
x_resh = tf.reshape(x_image, [my_batch, 48, 48, 48, 32])  # N_a_a_a_C
unit1_pool = max_pool_2x2x2(x_resh)  # 24*24*24*32
'''CONV3D LAYER2'''
unit2_conv_W = weight_variable([5, 5, 5, 32, 64], name="unit2_conv_W")
unit2_conv_B = bias_variable([64], name="unit2_conv_B")
unit2_conv_H = tf.nn.relu(conv2d(unit1_pool, unit2_conv_W) + unit2_conv_B)
'''POOL LAYER3'''
unit3_pool = max_pool_2x2x2(unit2_conv_H)  # 12*12*12*64
'''CONV3D LAYER4'''
unit4_conv_W = weight_variable([5, 5, 5, 64, 128], name="unit4_conv_W")
unit4_conv_B = bias_variable([128], name="unit4_conv_B")
unit4_conv_H = tf.nn.relu(conv2d(unit3_pool, unit4_conv_W) + unit4_conv_B)
'''POOL LAYER5'''
unit5_pool = max_pool_2x2x2(unit4_conv_H)  # 6*6*6*128
'''CONV3D LAYER6'''
unit6_conv_W = weight_variable([5, 5, 5, 128, 32], name="unit6_conv_W")
unit6_conv_B = bias_variable([32], name="unit6_conv_B")
unit6_conv_H = tf.nn.relu(conv2d(unit5_pool, unit6_conv_W) + unit6_conv_B)
'''FULL LAYER7'''
unit7_full_flat = tf.reshape(unit6_conv_H, shape=[-1, 6 * 6 * 6 * 32])
unit7_full_W = weight_variable([6 * 6 * 6 * 32, 2], name="unit7_full_W")
unit7_full_B = bias_variable([2], name="unit7_full_B")
y_conv = tf.nn.softmax(tf.matmul(unit7_full_flat, unit7_full_W) + unit7_full_B)
y_ = tf.placeholder(tf.float32, [my_batch, 2])

'''MODEL TRAINING OPS OBJECTS'''
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4, ).minimize(cross_entropy)
correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))

'''ADD ops to save and restore all the variables.'''
saver = tf.train.Saver()
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    saver.restore(sess, "tmpCNN3D/model.ckpt")
    print("Model restored.")
    data_test, label_test = dataset2.nextbatch()
    score = y_conv.eval(feed_dict={x_image: data_test, y_: label_test})
    print("predict score of proteins:")
    print(score)