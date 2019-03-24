#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np

import renom as rm
import renom.graph as rmg

rm.set_cuda_active(rm.has_cuda())

mnist = rm.utility.get_mnist(onehot=True, verbose=True)

X = mnist[0]
y = mnist[1]
print(y.shape)

X = X.astype(np.float32)
X /= X.max()
X = X.reshape(-1, 28 * 28)

model = rmg.Sequential([
    rmg.Dense(1000),
    rmg.Relu(),
    rmg.Dense(1000),
    rmg.Relu(),
    rmg.Dense(1000),
    rmg.Relu(),
    rmg.Dense(10),
])


epochs = 10
batch = 32

opt = rmg.Sgd()
x_in, y_in = rmg.Distro(X, y, batch_size=batch, test_split=0.9).get_output_graphs()
loss = rmg.SoftmaxCrossEntropy()
exe = loss(model(x_in), y_in).get_executor(mode='training', with_validation=True)

exe.execute(epochs)

#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))
