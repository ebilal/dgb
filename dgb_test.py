#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 12:52:31 2018

@author: ebilal
"""

import numpy as np
np.random.seed(1337)
from dgb import Dense, predict, accuracy, train, predict, batch_predict
from keras.datasets import mnist, cifar10
from keras.utils import np_utils


# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
#Y_train[Y_train == 0] = -1
#Y_test[Y_test == 0] = -1

if X_train.shape[-1] == 3:
	X_train = np.swapaxes(X_train, 2, 3)
	X_train = np.swapaxes(X_train, 1, 2)
	X_test = np.swapaxes(X_test, 2, 3)
	X_test = np.swapaxes(X_test, 1, 2)

X_train = np.reshape(X_train, (X_train.shape[0],-1))
X_test = np.reshape(X_test, (X_test.shape[0],-1))

X_test = (X_test - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 0.001)
X_train = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 0.001)

l1 = Dense(X_train.shape[1], 200, activation='relu')
l2 = Dense(l1.output_dim, 200, activation='relu')
l3 = Dense(l2.output_dim, 200, activation='relu')
l4 = Dense(l3.output_dim, 200, activation='relu')
l5 = Dense(l4.output_dim, 200, activation='relu')
l6 = Dense(l5.output_dim, 200, activation='relu')
l7 = Dense(l6.output_dim, 10, activation='softmax')

net = [l1, l2, l3, l4, l5, l6, l7]
train(net, X_train, Y_train, batch_size=100, epochs=10, lr=0.01, momentum=0.9, decay=0.0, verbose=1000, l2_reg=1)
(pred_train, loss, acc) = predict(net, X_train, Y_train)
print('Train acc: {} - loss: {}'.format(acc, loss))
(pred_test, loss, acc) = predict(net, X_test, Y_test)
print('Test acc: {} - loss: {}'.format(acc, loss))
