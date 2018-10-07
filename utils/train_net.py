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


def get_class_number(name):
    res = re.findall('[0-9]+', name)

    return int(res[0]) - 1


images = tf.placeholder(tf.float32, [None, 224, 224, 3])
net = MyNet({'data': images})
fc8 = net.layers['fc7']
init = tf.global_variables_initializer()
global_step = tf.Variable(0, trainable=False)

base = './resized_final_left_hand_data/'

names = [base + name for name in os.listdir(base)]
raw_name = [name for name in os.listdir(base)]
print(raw_name)
raw_name.sort()
names.sort()

epoch = 2
batch_size = 20
save_path = '../log/vgg16-1/ckpt/vgg16-fine_tune.ckpt'
log_path = '../log/summaries/m2/'

step = 0
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(log_path, sess.graph)

    sess.run(init)
    net.load('./mynet.npy', sess)

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
