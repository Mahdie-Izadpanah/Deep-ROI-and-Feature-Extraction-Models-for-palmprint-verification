import numpy as np
import tensorflow as tf
import os
import matplotlib.image as mp
import re
import pandas as pd
from termcolor import colored
from mynet import VGG_CNN_F as MyNet
import matplotlib.pyplot as plt
from math import sqrt

pic_dim = 224
class_dim = 4


def get_class(name):
    Chot = np.array([0.0] * class_dim)
    res = re.findall('[0-9]+', name)
    Chot[int(res[0]) - 1] = 1.0
    return Chot


def draw_box(image, out):
    points = [out[0] - out[2] / 2, out[1] - out[3] / 2, out[0] + out[2] / 2, out[1] + out[3] / 2]
    plt.imshow(image, cmap='gray')
    plt.plot([points[0], points[2]], [points[1], points[1]], c='g')
    plt.plot([points[2], points[2]], [points[1], points[3]], c='g')
    plt.plot([points[2], points[0]], [points[3], points[3]], c='g')
    plt.plot([points[0], points[0]], [points[3], points[1]], c='g')
    plt.show()


def get_kernel(name):
    for var in tf.trainable_variables():
        if name in var.name:
            return var


def get_class_number(name):
    res = re.findall('[0-9]+', name)
    return int(res[0]) - 1


def put_kernels_on_grid(kernel, pad=1):
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return i, int(n / i)

    (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    x = tf.transpose(x, (3, 0, 1, 2))
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    x = tf.transpose(x, (0, 2, 1, 3))
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    x = tf.transpose(x, (2, 1, 3, 0))

    x = tf.transpose(x, (3, 0, 1, 2))

    return x


def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


images = tf.placeholder(tf.float32, [None, 224, 224, 3])

# images = tf.div(tf.subtract(images, tf.reduce_mean(images)), reduce_std(images))

labels = tf.placeholder(tf.float32, [None, 4])

net = MyNet({'data': images})

fc8 = net.layers['fc7']

w = tf.Variable(tf.random_normal(shape=[4096, 4], mean=0, stddev=0.01), name='last_W')

b = tf.Variable(tf.zeros([4]), name='last_b')

output = tf.matmul(fc8, w) + b

cx = output[0, 0]
cy = output[0, 1]
w = output[0, 2]
h = output[0, 3]

xmin, xmax, ymin, ymax = cx - w / 2, cy + w / 2, cy - h / 2, cy + h / 2

image_with_box = tf.image.draw_bounding_boxes([images[0]], [[[ymin / 224, xmin / 224, ymax / 224, xmax / 224]]])

tf.summary.image('box', image_with_box)
cost = tf.reduce_mean(tf.abs(tf.subtract(output, labels)))
tf.summary.scalar('Cost', cost)

with tf.variable_scope('conv1'):
    tf.get_variable_scope().reuse_variables()
    weights1 = tf.get_variable('weights')
    grid1 = put_kernels_on_grid(weights1)
    tf.summary.image('conv1/kernels', grid1, max_outputs=3)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.96, staircase=True)

vars = [var for var in tf.trainable_variables() if 'last' in var.name]
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

merged = tf.summary.merge_all()
saver = tf.train.Saver()
init = tf.initialize_all_variables()

base = './org_data/'
boxes = pd.read_csv('NData.csv')

names = [base + name for name in os.listdir(base) if name in boxes['Name'].values]
raw_name = [name for name in os.listdir(base) if name in boxes['Name'].values]
raw_name.sort()
names.sort()

epoch = 100
batch_size = 20
save_path = '../log/vgg16-1/ckpt/vgg16_'
log_path = '../log/s/'
test_names = ['./left_hand/' + name for name in os.listdir('./left_hand/')]

bdata = boxes
bdata = bdata.set_index('Name')
step = 0
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(log_path, sess.graph)

    sess.run(init)
    net.load('./mynet.npy', sess)
    # saver.restore(sess, save_path + str(14400) + '.ckpt')

    for e in range(epoch):

        for i in range(int(len(names) / batch_size)):
            ims = [mp.imread(names[j]).reshape((pic_dim, pic_dim, 3)) for j in
                   range(i * batch_size, (i + 1) * batch_size)]
            label = []
            for name in range(i * batch_size, (i + 1) * batch_size):
                p = bdata.loc[raw_name[name]]
                cx = (p[0] + p[2]) / 2
                cy = (p[1] + p[3]) / 2
                w = abs(p[0] - p[2])
                h = abs(p[1] - p[3])
                label.append([cx, cy, w, h])

            _, c = sess.run([train_op, cost], feed_dict={images: ims, labels: label})
            print colored('Cost:', 'green'), colored(c, 'blue')

            if step % 2 == 0:
                import random

                r = random.randint(0, len(test_names) - 21)
                test = test_names[r:r + batch_size]
                image = [mp.imread(name).reshape((pic_dim, pic_dim, 3)) for name in test]
                summary = sess.run(merged, feed_dict={images: image, labels: label})
                summary_writer.add_summary(summary, step)

                saver.save(sess, save_path + str(step) + '.ckpt')
                print 'saved step : ', step

            step += 1