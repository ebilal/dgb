#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:03:04 2018

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
        self.W = np.random.normal(0., np.sqrt(1./input_dim), (input_dim, output_dim))
        self.b = np.random.normal(0., np.sqrt(1./input_dim), (output_dim,))
#        self.W = np.random.uniform(-np.sqrt(1.7/input_dim), np.sqrt(1.7/input_dim), (input_dim, output_dim))
#        self.b = np.random.uniform(-np.sqrt(1.7/input_dim), np.sqrt(1.7/input_dim), (output_dim,))

        # Output before nonlinearity
        self.Z = None
        # Output after nonlinearity
        self.A = None
        # Pseudo-residual
        self.dZ = None
        # Gradient / Velocity
        self.gW = np.zeros((input_dim, output_dim))
        self.gb = np.zeros((output_dim,))

    def fw_activation(self, x, name=None):
        if name is None:
            name = self.activation

        if name == 'sigmoid':
            out = expit(x)
        elif name == 'softplus':
            out = x
            out[x<20] = np.log(1. + np.exp(x[x<20]))
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
        out = self.fw_activation(np.dot(X, self.W) + self.b)

        return out


def fw(net, X, y):
    n = len(net)

    for k in range(n):
        net[k].Z = np.dot(X, net[k].W) + net[k].b
        net[k].A = net[k].fw_activation(net[k].Z)
        X = net[k].A

    pred = X
    acc = accuracy(pred, y)
    if pred.ndim > 1 and pred.shape[1] > 1:
        loss = metrics.log_loss(np.argmax(y, axis=1), pred, labels=range(pred.shape[1]))
    else:
        loss = metrics.log_loss(y, pred, labels=[0,1])

    return (pred, loss, acc)


def residual_gradient(Z, dZ, W, activation):

    if activation == 'sigmoid':
        A = expit(Z)
        dA = np.multiply(A, 1. - A)
    elif activation == 'softplus':
        dA = expit(Z)
    elif activation == 'tanh':
        dA = 1. - np.square(np.tanh(Z))
    elif activation == 'relu':
        dA = np.zeros(Z.shape, dtype=np.float64)
        dA[Z > 0.] = 1.
    elif activation == 'linear':
        dA = 1.

    grad = np.multiply(dA, np.dot(dZ, np.transpose(W)))
    grad = clip_norm(grad, 1.)

    return grad


def weight_gradient(A, dZ, l2_reg):

    clf = Ridge(alpha=l2_reg)
    clf.fit(A, dZ)
    dW = np.transpose(clf.coef_)
    db = clf.intercept_

    #db = np.clip(db, -0.1, 0.1)
    #dW = np.clip(dW, -0.1, 0.1)

    return (dW, db)


def fwbkfw(net, X_train, y_train, lr, momentum, l2_reg):

    n = len(net)

    # Forward pass
    pred, loss, acc = fw(net, X_train, y_train)

    # Backward pass
    net[n-1].dZ = (pred - y_train)
    for k in reversed(xrange(n-1)):
        net[k].dZ = residual_gradient(net[k].Z, net[k+1].dZ, net[k+1].W, net[k].activation)

    # Forward pass
    dW, db = weight_gradient(X_train, net[0].dZ, l2_reg)
    net[0].gW = momentum * net[0].gW - lr * dW
    net[0].gb = momentum * net[0].gb - lr * db
    net[0].W = net[0].W + net[0].gW
    net[0].b = net[0].b + net[0].gb
    for k in xrange(1, n):
        dW, db = weight_gradient(net[k-1].A, net[k].dZ, l2_reg)
        net[k].gW = momentum * net[k].gW - lr * dW
        net[k].gb = momentum * net[k].gb - lr * db
        net[k].W = net[k].W + net[k].gW
        net[k].b = net[k].b + net[k].gb

    return (loss, acc)


def train(net, X, y, batch_size, epochs, lr, momentum, decay, l2_reg=1, verbose=0):

    n, m = X.shape

    running_loss = 0
    running_acc = 0
    for e in xrange(epochs):

        ind = np.arange(n)
        np.random.shuffle(ind)
        X = X[ind]
        y = y[ind]

        for i in xrange(0, n, batch_size):
            X_train = X[i:i + batch_size]
            y_train = y[i:i + batch_size]
            loss, acc = fwbkfw(net, X_train, y_train, lr, momentum, l2_reg)

            if running_loss > 0 and running_acc > 0:
                running_loss = 0.9 * running_loss + 0.1 * loss
                running_acc = 0.9 * running_acc + 0.1 * acc
            else:
                running_loss = loss
                running_acc = acc

            batch_num = int(i / batch_size) + 1
            if verbose > 0 and batch_num % verbose == 0:
                print('Epoch %d - Batch %d/%d - loss: %.4f - acc: %.2f' %(e, batch_num, int(n/batch_size), running_loss, running_acc))

        if isinstance(decay, list):
            for ix, d in enumerate(decay):
                if e == d:
                    lr = 0.1 * lr
                    break
        else:
            lr = lr * (1. / (1. + decay * e))


def predict(net, X, y=None):
    n = len(net)

    for k in xrange(n):
        pred = net[k].get_output(X)
        X = pred

    if y is not None:
        if pred.ndim > 1 and pred.shape[1] > 1:
            loss = metrics.log_loss(np.argmax(y, axis=1), pred, labels=range(pred.shape[1]))
        else:
            loss = metrics.log_loss(y, pred, labels=[0,1])
        acc = accuracy(pred, y)

        return (pred, loss, acc)
    else:
        return pred


def batch_predict(net, X, batch_size=1000):
    pred = []
    for i in xrange(0, X.shape[0], batch_size):
        pred.append(predict(net, X[i:i+batch_size,:]))

    pred = np.vstack(pred)
    return pred


def accuracy(x, y):
    if x.ndim > 1 and x.shape[1] > 1:
        acc = metrics.accuracy_score(np.argmax(x, axis=1), np.argmax(y, axis=1))
    else:
        acc = metrics.accuracy_score(np.round(x), np.round(y))

    return acc

def clip_norm(dZ, c):
    n, m = dZ.shape

    nrm = np.linalg.norm(dZ, axis=0)
    for k in range(m):
        if nrm[k] > c:
            dZ[:,k] = dZ[:,k] * c / nrm[k]

    return dZ

