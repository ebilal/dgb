#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 12:16:02 2018

@author: ebilal
"""

from __future__ import print_function
from __future__ import division

import numpy as np
np.random.seed(1337)
from scipy.special import expit
from sklearn import metrics
from sklearn.linear_model import Ridge



class Dense:
    def __init__(self, input_dim, output_dim, activation):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        # Initialize weights
        self.W = np.random.normal(0., 1., (input_dim, output_dim))

        self.Z = None
        self.dZ = None

        self.gW = np.zeros((input_dim, output_dim))

    def fw_activation(self, x, name=None):
        if name is None:
            name = self.activation

        if name == 'sigmoid':
            out = expit(x)
        elif name == 'softplus':
            out = x
            out[x<20] = np.log(1 + np.exp(x[x<20]))
        elif name == 'relu':
            out = np.maximum(x, 0, x)
        elif name == 'tanh':
            out = np.tanh(x)
        elif name == 'linear':
            out = x
        elif name == 'softmax':
            exps = np.exp(x - np.max(x, axis=1, keepdims=True))
            out = exps / np.sum(exps, axis=1, keepdims=True)

        return out

    def get_output(self, X):
        out = self.fw_activation(np.dot(X, self.W))

        return out


def fw(net, X, y):
    n = len(net)

    for k in range(n):
        net[k].Z = np.dot(X, net[k].W)
        X = net[k].fw_activation(net[k].Z)

    pred = X
    loss = metrics.mean_squared_error(y, pred)

    return (pred, loss)


def residual_gradient(Z, dZ, W, activation):

    if activation == 'sigmoid':
        A = expit(Z)
        dA = np.multiply(A, 1 - A)
    elif activation == 'softplus':
        dA = expit(Z)
    elif activation == 'tanh':
        dA = 1 - np.square(np.tanh(Z))
    elif activation == 'relu':
        dA = np.zeros(Z.shape)
        dA[Z > 0] = 1.
    elif activation == 'linear':
        dA = 1. * np.ones(Z.shape)

    grad = np.multiply(dA, np.dot(dZ, np.transpose(W)))
    grad = np.clip(grad, -0.1, 0.1)

    return grad


def weight_gradient(A, dZ, l2_reg):

    clf = Ridge(alpha=l2_reg, fit_intercept=False)
    clf.fit(A, dZ)
    dW = np.transpose(clf.coef_)

    return dW


def fwbkfw(net, X_train, y_train, lr, momentum, l2_reg):

    n = len(net)

    # Forward pass
    pred, loss = fw(net, X_train, y_train)

    # Backward pass
    net[n-1].dZ = 2*(pred - y_train)
    for k in reversed(xrange(n-1)):
        res_grad = residual_gradient(net[k].Z, net[k+1].dZ, net[k+1].W, net[k].activation)
        net[k].dZ = res_grad

    # Forward pass
    for k in xrange(n):
        if k == 0:
            A = X_train
        else:
            A = net[k-1].fw_activation(net[k-1].Z)
        dW = weight_gradient(A, net[k].dZ, l2_reg)
        net[k].gW = momentum * net[k].gW - lr * dW
        net[k].W = net[k].W + net[k].gW

    return loss


def train(net, X, y, batch_size, epochs, lr, momentum, decay, l2_reg=1, verbose=0):

    n, m = X.shape
    running_loss = 0
    for e in xrange(epochs):
        ind = np.arange(n)
        np.random.shuffle(ind)
        X = X[ind]
        y = y[ind]
        for i in xrange(0, n, batch_size):
            X_train = X[i:i + batch_size]
            y_train = y[i:i + batch_size]
            loss = fwbkfw(net, X_train, y_train, lr, momentum, l2_reg)
            if running_loss > 0:
                running_loss = 0.9 * running_loss + 0.1 * loss
            else:
                running_loss = loss

            if verbose > 0 and i % verbose == 0:
                print('Epoch %d - Batch %d - loss: %.4f' %(e, i, running_loss))

        lr = lr * (1. / (1. + decay * e))


def predict(net, X, y=None):
    n = len(net)

    for k in xrange(n):
        pred = net[k].get_output(X)
        X = pred

    if y is not None:
        loss = metrics.mean_squared_error(y, pred)

        return (pred, loss)
    else:
        return pred


def batch_predict(net, X, batch_size=1000):
    pred = []
    for i in xrange(0, X.shape[0], batch_size):
        pred.append(predict(net, X[i:i+batch_size,:]))

    pred = np.vstack(pred)
    return pred

