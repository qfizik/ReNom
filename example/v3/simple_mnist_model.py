#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np

import renom as rm
import renom.graph as rmg
from renom.auxiliary.mnist import get_mnist

rm.set_cuda_active(rm.has_cuda())

mnist = get_mnist(onehot=True, verbose=True)

X = mnist[0]
y = mnist[1]
print(y.shape)

X = X.astype(np.float32)
X /= X.max()
X = X.reshape(-1, 28 * 28)

model = rmg.Sequential([
    rmg.Dense(2000, ignore_bias=True),
    rmg.Relu(),
    rmg.Dense(10, ignore_bias=True),
])


epochs = 10

opt = rmg.Rmsprop()
x_in, y_in = rmg.DataInput([X, y]).index().batch(1024).get_output_graphs()
loss = rmg.SoftmaxCrossEntropy()
exe = loss(model(x_in), y_in).get_executor(mode='training', optimizer=opt)

exe.execute(epochs=epochs)

#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))
