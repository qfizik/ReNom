import time
start_time = time.time()
import renom.cuda as cu
import faulthandler
import numpy as np
import renom as rm
from renom.config import precision
from renom.cuda.gpuvalue import *
from renom.utility.distributor.distributor import NdarrayDistributor, GPUDistributor
from renom.cuda import set_cuda_active
from renom.cuda.base.cuda_base import queryDeviceProperties, cuDeviceSynchronize
import h5py
from sklearn.preprocessing import LabelBinarizer
from renom.utility.trainer import Trainer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from renom.core import Node
import sys
import argparse

faulthandler.enable()
queryDeviceProperties()

parser = argparse.ArgumentParser()
parser.add_argument('-np','--numpy_distributor',help='Selects the NdarrayDistributor as the distributor.', action='store_true')
parser.add_argument('-c','--use_cuda',help='Activates cuda.', action='store_false')
parser.add_argument('-e','--epochs',help='Determines the number of epochs to be run.',type=int,nargs='?',default=1)
parser.add_argument('-b','--batch_size',help='Sets the batch size to be used.',type=int,nargs='?',default=64)
parser.add_argument('-sc','--score',help='Displays the final evaluation of the tested model',action='store_true')
parser.add_argument('-dm','--dense_model',help='Choose the alternative dense model.',action='store_true')
parser.add_argument('-rm','--recurrent_model',help='Choose the alternative recurrent model.', action='store_true')
parser.add_argument('-r','--repeats',help='Run the training procedure <arg> amount of times',type=int,nargs='?',default=1)
parser.add_argument('-g','--gpus',help='Determines number of GPUs to use',type=int,nargs='?',default=1)
args = parser.parse_args()


with h5py.File('mnist-dataset.hdf5','r') as f:
    X = np.array(f['training-set']['Images'])
    Y = np.array(f['training-set']['Labels'])
X = X / np.amax(X).astype(precision)
Y = LabelBinarizer().fit_transform(Y).astype(precision)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1)

#X = np.ones((int(1e5),8,32,32))
#Y = np.ones((int(1e5),1))
#X_train = X
#Y_train = Y

class ReshapeModel(rm.Model):
    def __init__(self, new_shape):
        self._new_shape = list(new_shape)

    def forward(self, x):
        return x.reshape(x.shape[0], *self._new_shape)

np.set_printoptions(precision=4)

class StopModel(rm.Model):
    def __init__(self, should_stop=False, name = None):
        self._should_stop = should_stop
        self._name = name

    def forward(self, x):
        x.to_cpu()
        if self._name is not None:
            print("Printing for {}".format(self._name))
        print(x.ravel()[0:10])
        if self._should_stop:
            assert False
        else:
            return x

class ChainedModel(rm.Lstm):
    def __init__(self, *args, **kwargs):
        super(ChainedModel, self).__init__(*args, **kwargs)

    def forward(self, x):
        lstm_model = super(ChainedModel, self)
        lstm_model.truncate()
        length = x.shape[1]
        for i in range(length):
            ret = lstm_model.forward(x[:, i])
        return ret

mnist_model = rm.Sequential([
        ReshapeModel((1, 28, 28)),
        rm.Conv2d(channel=32,filter=5,padding=2),
        rm.Relu(),
        rm.MaxPool2d(filter=2,stride=2),
        rm.Conv2d(channel=64,filter=5,padding=2),
        rm.Relu(),
        rm.MaxPool2d(filter=2,stride=2),
        ReshapeModel([-1]),
        rm.Dense(1024),
        rm.Dense(10),
])

dnn_model = rm.Sequential([
        ReshapeModel(([-1])),
        rm.Dense(1000),
        rm.Dense(100),
        rm.Dense(10)
])

lstm_model = rm.Sequential([
        ReshapeModel((28, 28,)),
        #rm.ChainedLSTM(50),
        ChainedModel(50),
        rm.Dense(10),
])

print ("Using Cuda: \033[1;3{1}m{0}\033[0m".format(args.use_cuda,2 if args.use_cuda else 1))
using_streams = set_cuda_active.__code__.co_argcount == 2
print("{0}Using streams".format("\033[1;31mNOT\033[0m " if not using_streams else ""))
if using_streams:
    set_cuda_active(args.use_cuda,True)
else:
    set_cuda_active(args.use_cuda)
distributor_type = GPUDistributor if args.use_cuda and not args.numpy_distributor else NdarrayDistributor
print("Using distributor: \033[1;33m{0}\033[0m".format(distributor_type))
distributor = distributor_type(X_train,Y_train,gpus=args.gpus)
#test_distributor = distributor_type(X_test, Y_test)

batch_size = args.batch_size
print("One epoch equals {:d} steps".format(len(X_train)//batch_size))
epochs = args.epochs
print("Running {:d} epochs".format(epochs))

#opt = rm.Adadelta()
#opt = rm.Adamax()
opt = rm.Sgd(0.001)

#GPUDistributor.LOW_MEMORY(True)

import benchmarker as bm

model = []

if args.dense_model:
    print("Using Dense model")
    model = dnn_model
elif args.recurrent_model:
    print("Using LSTM model")
    model = lstm_model
else:
    print("Using Normal model")
    model = mnist_model
    #for i in range(args.gpus):
    #    with rm.cuda.RenomHandler(i): pass
    #    model.append(rm.Sequential([
    #        ReshapeModel((1, 28, 28)),
    #        rm.Conv2d(channel=32,filter=5,padding=2,activation=rm.Relu()),
    #        rm.Conv2d(channel=32,filter=5,padding=2,activation=rm.Relu()),
    #        rm.Conv2d(channel=32,filter=5,padding=2,activation=rm.Relu()),
    #        rm.MaxPool2d(filter=2,stride=2),
    #        rm.Conv2d(channel=64,filter=5,padding=2,activation=rm.Relu()),
    #        rm.Conv2d(channel=64,filter=5,padding=2,activation=rm.Relu()),
    #        rm.Conv2d(channel=64,filter=5,padding=2,activation=rm.Relu()),
    #        rm.MaxPool2d(filter=2,stride=2),
    #        ReshapeModel([-1]),
    #        rm.Dense(1024),
    #        rm.Dense(10),
    #    ]))
    #    model[i].set_gpu(i)


#trainer = Trainer(model, num_epoch=epochs, loss_func = rm.softmax_cross_entropy, batch_size = args.batch_size, optimizer = opt)
bm.startTiming("Renom Main Loop")
#trainer.train(distributor)
for e in range(epochs):
    if e == 0:
        bm.startTiming("Warmup Epoch")
    elif e > 0:
        bm.newTiming("Normal Epoch")
    for batch, label in distributor.batch(batch_size):
        for i in range(args.gpus):
            if batch[i].shape[0] < 1:
                continue
            with cu.use_device(i):
                with model.train():
                    z = model(batch)
                    l = rm.softmax_cross_entropy(z, label)
                l.grad().update(opt)

#cuDeviceSynchronize()
bm.endAllTiming()
bm.getTimes()

if args.score:
    predictions = model(X_test[0:1000]).as_ndarray()
    #predictions = model(X_test[0:1000]).new_array()
    predictions = np.array(np.argmax(predictions,axis=1))
    label = np.array(np.argmax(Y_test[0:1000],axis=1))
    print(confusion_matrix(label, predictions))
    print(classification_report(label, predictions))
