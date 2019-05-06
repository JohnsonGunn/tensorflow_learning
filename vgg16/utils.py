#!/usr/bin/python
#coding:utf-8

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

#正常显示中文
mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS']
#正常显示正负号
mpl.rcParams['axes.unicode_minus'] = False

#对要处理的图像做一些预处理操作
def load_image(path):
    fig = plt.figure('Center and Resize')
    img = io.imread(path)
    #将像素点进行归一化处理，压缩到0-1之间
    img = img/255
    ax0 = fig.add_subplot(131)
    ax0.set_xlabel(u'Original Picture')
    ax0.imshow(img)

    short_edge = min(img.shape[0:2])
    y = (img.shape[0] - short_edge)/2
    x = (img.shape[1] - short_edge)/2
    crop_image = img[y:y+short_edge, x:x+short_edge]
    print(crop_image.shape)

    ax1 = fig.add_subplot(132)
    ax1.set_label(u'Center Picture')
    ax1.imshow(crop_image)

    re_img = transform.resize(crop_image, (224, 224))

    ax2 = fig.add_subplot(133)
    ax2.set_xlabel(u'Resize Picture')
    ax2.imshow(re_img)
    img_ready = re_img.reshape((1, 224, 224, 3))
    return img_ready

def percent(value):
    return '%.2f %%'%(value*100)