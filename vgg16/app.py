#!/usr/bin/python
#coding:utf-8
import numpy as np
import vgg16
import matplotlib
import utils
import tensorflow as tf
import matplotlib.pyplot as plt
from Nclasses import labels

img_path = input('Input the path and image name:')
#调用load_image()函数，对待测试的图片做一些处理
img_ready = utils.load_image(img_path)

fig = plt.figure(u'Top-5 预测结果')

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.forward(x)
    probability = sess.run(vgg.prob, feed_dict={x:img_ready})
    top5 = np.argsort(probability[0])[-1:-6:-1]
    print('Top-5:', top5)


