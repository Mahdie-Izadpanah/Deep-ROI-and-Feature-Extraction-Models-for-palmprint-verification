import numpy as np
import random
import tensorflow as tf
import os
import matplotlib.image as mp
import re
import pandas as pd
from termcolor import colored

from mynet import VGG_CNN_F as MyNet
import matplotlib.pyplot as plt

pic_dim = 224
class_dim = 386


def draw_box(base_dir, name, out):
    image = mp.imread(base_dir + name)
    points = [out[0] - out[2] / 2, out[1] - out[3] / 2, out[0] + out[2] / 2, out[1] + out[3] / 2]
    plt.imshow(image, cmap='gray')
    plt.plot([points[0], points[2]], [points[1], points[1]], c='g')
    plt.plot([points[2], points[2]], [points[1], points[3]], c='g')
    plt.plot([points[2], points[0]], [points[3], points[3]], c='g')
    plt.plot([points[0], points[0]], [points[3], points[1]], c='g')
    plt.show()

#reduce dimention estefade nakardim
def reduce_var(x, axis=None, keepdims=False):

    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keepdims)


def reduce_std(x, axis=None, keepdims=False):

    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def get_class(name):
    Chot = np.array([0.0] * class_dim)
    res = re.findall('[0-9]+', name)
    Chot[int(res[0]) - 1] = 1.0
    return Chot


def get_class_number(name):
    res = re.findall('[0-9]+', name)

    return int(res[0]) - 1


batch_size = 1

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
images = tf.div(tf.subtract(images, tf.reduce_mean(images)), reduce_std(images))

labels = tf.placeholder(tf.float32, [None, 386])
net = MyNet({'data': images})

fc8 = net.layers['fc7']

w = tf.Variable(tf.random_normal(shape=[4096, 386], mean=0, stddev=2 / np.sqrt(4096)), name='last_W')
b = tf.Variable(tf.zeros([386]), name='last_b')
output = tf.matmul(fc8, w) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output))
tf.summary.scalar('Cost', cost)
# opt = tf.train.AdamOptimizer(learning_rate=0.0001)
opt = tf.train.AdamOptimizer(learning_rate=0.00001)
vars = [var for var in tf.trainable_variables() if 'last' in var.name]
train_op = opt.minimize(cost)

merged = tf.summary.merge_all()
saver = tf.train.Saver()
init = tf.initialize_all_variables()

base = './resized_final_left_hand_data/'
# boxes = pd.read_csv('NData.csv')

names = [base + name for name in os.listdir(base)]
raw_name = [name for name in os.listdir(base)]
print(raw_name)
raw_name.sort()
names.sort()

epoch = 2
batch_size = 1
save_path = '../log/vgg16-1/ckpt/vgg16-fine_tune.ckpt'
log_path = '../log/summaries/m2/'

# bdata = boxes
# bdata = bdata.set_index('Name')
step = 0
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(log_path, sess.graph)

    sess.run(init)
    net.load('./mynet.npy', sess)
    # saver.restore(sess, save_path)
    # for e in range(epoch):
    #
    #     for i in range(int(len(names) / batch_size)):
    #         ims = [mp.imread(names[j]).reshape((pic_dim, pic_dim, 3)) for j in
    #                range(i * batch_size, (i + 1) * batch_size)]
    #         label = [get_class(names[j]) for j in range(i * batch_size, (i + 1) * batch_size)]
    #         _, c = sess.run([train_op, cost], feed_dict={images: ims, labels: label})
    #         print colored('Cost:', 'green'), colored(c, 'blue')
    #
    #         if step % 2 == 0:
    #             summary = sess.run(merged, feed_dict={images: ims, labels: label})
    #             summary_writer.add_summary(summary, step)
    #             saver.save(sess, save_path)
    #             print('saved.')
    features = []
    for i in range(int(len(names) / batch_size)):
        ims = [mp.imread(names[j]).reshape((pic_dim, pic_dim, 3)) for j in
               range(i * batch_size, (i + 1) * batch_size)]
        lbs = [get_class_number(raw_name[j]) for j in
               range(i * batch_size, (i + 1) * batch_size)]
        for lb, feature in zip(lbs, sess.run(fc8, feed_dict={images: ims})):
            features.append(list(feature) + [lb])

    df_features = pd.DataFrame(data=features, columns=[str(i) for i in range(4096)] + ['label'])
    print(df_features)
    df_features.to_csv('features.csv', index=False)

# with tf.Session() as sess:
#     summary_writer = tf.summary.FileWriter(log_path, sess.graph)
#
#     sess.run(init)
#     net.load('./mynet.npy', sess)
#     features = []
#     for i in range(int(len(names) / batch_size)):
#         ims = [mp.imread(names[j]).reshape((pic_dim, pic_dim, 3)) for j in
#                range(i * batch_size, (i + 1) * batch_size)]
#
#         for j, feature in enumerate(sess.run(fc8, feed_dict={images: ims})):
#             features.append(list(feature) + [raw_name[j]])
#     df_features = pd.DataFrame(data=features,columns=[str(i) for i in range(4096)] + ['label'])
#     print(df_features)
#     df_features.to_csv('data.csv',index=False)
