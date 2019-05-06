#!/usr/bin/python

#coding:utf-8

'''

从下载的vgg16模型中读出网络的参数
模型是vgg16.npy，是一个字典数据，可以通过np.load读出

'''
import inspect
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

#训练样本的平均值
VGG_MEAN = [103.939, 116.779, 123.68]

class Vgg16():
    def __init__(self, vgg16_path=None):
        if vgg16_path is None:
            #os.getcwd()用于返回当前工作目录
            vgg16_path = os.path.join(os.getcwd(), 'vgg16.npy')
            print(vgg16_path)

            #遍历这个字典中的键值对，导入模型参数
            self.data_dict = np.load(vgg16_path, encoding='latin1').item()
        for x in self.data_dict:
            print(x)

    def forward(self, images):
        print('Build model start!')
        #获取前向传播开始时间
        start_time = time.time()
        #逐像素乘以255
        rgb_scaled = images * 255
        #从GRB通道转换成BGR通道
        red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]

        #减去每个通道的像素平均值，这样可以移除图像的平均亮度
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2]
        ])

        assert bgr.get_shape().as_list()[1:] == [224, 224, 1]

        #下面搭建VGG16卷积神经网络

        #第一个卷积，两个卷积层和一个最大池化层
        self.conv1_1 = self.conv_layer(bgr, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
        self.pool1 = self.max_pool_2x2(self.conv1_2, 'pool1')

        #第二个卷积，两个卷积层和一个最大池化层
        self.conv2_1 = self.conv_layer(self.pool1, 'con2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
        self.pool2 = self.max_pool_2x2(self.conv2_2, 'pool2')

        #第三个卷积，三个卷积层和一个最大池化层
        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
        self.pool3 = self.max_pool_2x2(self.conv3_3, 'pool3')

        #第四个卷积，三个卷积层和一个最大池化层
        self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
        self.pool4 = self.max_pool_2x2(self.conv4_3, 'pool4')

        #第五个卷积，三个卷积层和一个最大池化层
        self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
        self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
        self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
        self.pool4 = self.conv_layer(self.conv5_3, 'pool5')

        #第六层全连接层
        self.fc6 = self.fc_layer(self.pool4, 'fc6')
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        #第七层全连接层
        self.fc7 = self.fc_layer(self.fc6, 'fc7')
        self.relu7 = tf.nn.relu(self.fc7)

        #第七层全连接层
        self.fc8 = self.fc_layer(self.fc7, 'fc8')
        self.prob = tf.nn.softmax(self.fc8, name='prob')

        end_time = time.time()
        print('Time consuming: %f'%(end_time-start_time))

        #将本次读到的模型参数清0
        self.data_dict = None

    def conv_layer(self, x, name):
        with tf.variable_scope(name):
            #读出该层的卷积核参数
            w = self.get_conv_filter(name)
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
            #读出该层的偏置
            conv_biases = self.get_bias(name)
            result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))
            return result

    def max_pool_2x2(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def fc_layer(self, x, name):
        with tf.variable_scope(name):
            shape = x.get_shape().as_list()
            print('Fully_connected_layer_shape:', shape)
            dim = 1
            for i in shape[1:]:
                dim *= i
            x = tf.reshape(x, [-1, dim])
            w = self.get_fc_weights(name)
            b = self.get_bias(name)
            result = tf.nn.bias_add(tf.matmul(x, w), b)
            return result

    #从模型字典中读出各卷积层卷积核的参数
    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name='filter')


    #从模型字典中读出各卷积层偏置参数
    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='biases')

    #从模型字典中读出全连接权重参数
    def get_fc_weights(self, name):
        return tf.constant(self.data_dict[name][0], name='weights')




