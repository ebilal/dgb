#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:05:27 2018

@author: ebilal
"""

import numpy as np
from ali import Dense, predict, train, predict, batch_predict



xdim=6
wdim=6
ydim=100
nsamples=1000
A_condition_number = 1e-20

# the true map from x to y
Atrue = np.linspace(1, A_condition_number, ydim).reshape(-1, 1) * np.random.rand(ydim, xdim)

# the inputs
X = np.random.randn(xdim, nsamples)
# the y's to fit
Ytrue = np.dot(Atrue, X)

X = np.transpose(X)
Ytrue = np.transpose(Ytrue)

l1 = Dense(X.shape[1], wdim, activation='linear')
l2 = Dense(wdim, ydim, activation='linear')

net = [l1, l2]
train(net, X, Ytrue, batch_size=100, epochs=100, lr=0.01, momentum=0.0, decay=0.0, verbose=1000, l2_reg=0)
(pred_train, loss) = predict(net, X, Ytrue)
print('Train loss: {}'.format(loss))

