#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This is the example codes for PE-LTR.
Created on Wed Oct  9 11:27:53 2019

@author: jackielinxiao
"""

import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(2, 100)) 
y_data = np.dot([0.100, 0.200], x_data) + 0.300

weight_a = tf.placeholder(tf.float32)
weight_b = tf.placeholder(tf.float32)

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

loss_a = tf.reduce_mean(tf.square(y - y_data))
loss_b = tf.reduce_mean(tf.square(W))
loss = weight_a * loss_a + weight_b * loss_b    

optimizer = tf.train.GradientDescentOptimizer(0.5)

a_gradients = tf.gradients(loss_a, W)
b_gradients = tf.gradients(loss_b, W)

train = optimizer.minimize(loss)


init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

def pareto_step(weights_list, out_gradients_list):
    model_gradients = out_gradients_list
    M1 = model_gradients*np.transpose(model_gradients)
    e = np.mat(np.ones(np.shape(weights_list)))
    M = np.hstack((M1,np.transpose(e)))
    mid = np.hstack((e,np.mat(np.zeros((1,1)))))
    M = np.vstack((M,mid))
    z = np.mat(np.zeros(np.shape(weights_list)))
    nid = np.hstack((z,np.mat(np.ones((1,1)))))
    w = M*np.linalg.inv(M*np.transpose(M))*np.transpose(nid)
    if len(w)>1:
        w = np.transpose(w)
        w = w[0,0:np.shape(w)[1]]
        mid = np.where(w > 0, 1.0, 0)
        nid = np.multiply(mid, w)
        uid = sorted(nid[0].tolist()[0], reverse=True)
        sv = np.cumsum(uid)
        rho = np.where(uid > (sv - 1.0) / range(1,len(uid)+1), 1.0, 0.0)
        r = max(np.argwhere(rho))
        theta = max(0, (sv[r] - 1.0) / (r+1))
        w = np.where(nid - theta>0.0, nid - theta, 0)
    return w

w_a = 0.5
w_b = 0.5
for step in xrange(0, 20):
    res = sess.run([a_gradients,b_gradients,train],feed_dict={weight_a:w_a,weight_b:w_b})
    #res[0]
    weights = np.mat([w_a, w_b])
    paras = np.vstack((res[0][0],res[1][0]))
    mid = pareto_step(weights,paras)
    w_a, w_b = mid[0,0], mid[0,1]
    print w_a, w_b, step, sess.run(W), sess.run(b)