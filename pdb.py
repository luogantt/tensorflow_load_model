#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:15:53 2019

@author: lg
"""

#coding=utf-8 
from __future__ import absolute_import, unicode_literals
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework.graph_util import convert_variables_to_constants 
from tensorflow.python.framework import graph_util 
import cv2 
import numpy as np  
mnist = input_data.read_data_sets(".",one_hot = True)
import tensorflow as tf
 
#用于将自定义输入图片反转
def reversePic(src):
        # 图像反转  
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src[i,j] = 255 - src[i,j]
    return src 
 
with tf.Session() as persisted_sess:
  print("load graph")
  with tf.gfile.FastGFile("grf.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
  # print("map variables")
  with tf.Session() as sess:
 
        # tf.initialize_all_variables().run()
        input_x = sess.graph.get_tensor_by_name("Mul:0")
        y_conv_2 = sess.graph.get_tensor_by_name("final_result:0")
 
 
        path="pic/e2.jpg"  
        im = cv2.imread(path,cv2.IMREAD_GRAYSCALE) 
        #反转图像，因为e2.jpg为白底黑字   
        im =reversePic(im)
#        cv2.namedWindow("camera", cv2.WINDOW_NORMAL); 
#        cv2.imshow('camera',im)  
#        cv2.waitKey(0) 
        # im=cv2.threshold(im, , 255, cv2.THRESH_BINARY_INV)[1];
 
        im = cv2.resize(im,(28,28),interpolation=cv2.INTER_CUBIC)  
        # im =reversePic(im)
        # im=cv2.threshold(im,200,255,cv2.THRESH_TRUNC)[1]
        # im=cv2.threshold(im,60,255,cv2.THRESH_TOZERO)[1]
 
        # img_gray = (im - (255 / 2.0)) / 255  
        x_img = np.reshape(im , [-1 , 784])  
        output = sess.run(y_conv_2 , feed_dict={input_x:x_img})  
        print ('the predict is %d' % (np.argmax(output)) )
        #关闭会话  
        sess.close() 