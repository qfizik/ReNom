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
X = X.reshape(-1, 28, 28)

x_in, y_in = rmg.DataInput([X, y], num_gpus=1).shuffle().batch(64).get_output_graphs()


reg = rmg.L2()
model = rmg.Sequential([
    rmg.Lstm(10),
])
loss_func = rmg.SoftmaxCrossEntropy()

for i in range(28):
    out = model(x_in[:, :, i])

loss = loss_func(out, y_in)

epochs = 10

opt = rmg.Rmsprop()
exe = loss.get_executor(mode='training', optimizer=opt)

def reg_wd(info):
    if 'step' not in info:
        info['step'] = 0
    else:
        info['step'] += 1
    if info['step'] == 20:
        reg.wd = 0
exe.register_event('Step-Finish', reg_wd)

exe.execute(epochs=epochs)

#print(confusion_matrix(y_test, predictions))
#print(classification_report(y_test, predictions))
