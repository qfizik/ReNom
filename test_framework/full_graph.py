import numpy as np
import renom as rm
from convo2d import convo2d
from max_pool import max_pool
from dense import dense
from distributor import distributor
from softmax_loss import softmax_loss
import benchmarker as bm
from sklearn.preprocessing import LabelBinarizer
import h5py
rm.set_cuda_active()


graph_network = [ convo2d(channels = 32, kernel = 5, padding = 2),
                  max_pool(kernel = 2, stride = 2),
                  convo2d(channels = 64, kernel = 5, padding = 2),
                  max_pool(kernel = 2, stride = 2),
                  dense(out_size = 1024),
                  dense(out_size = 10),
                ]

with h5py.File('mnist-dataset.hdf5','r') as f:
  x = np.array(f['training-set']['Images'])
  y = np.array(f['training-set']['Labels'])
data = x / np.amax(x).astype(rm.precision)
labels = LabelBinarizer().fit_transform(y).astype(rm.precision)

if __name__ == '__main__' and False:

  lbls = np.zeros((1,2)).astype(rm.precision)
  lbls[0,1] = 1
  dist = distributor(data = np.ones((1,1)).astype(rm.precision), labels = lbls, batch_size = 1, num_gpus = 1)
  dist.setup()
  outs = dist._outputs

  den = dense(out_size = 2)
  den.setup(outs)
  outs = den._outputs
  
  soft = softmax_loss()
  soft.setup(outs)

  backs = dist._label_outputs
  soft.setup_backward(backs)
  backs = soft._backwards
  den.setup_backward(backs)

  dist.forward()
  den.forward()
  soft.forward()

  soft.backward()
  den.backward()
  print('Weights (back/forward)')
  print(den._weights_back[0].new_array())
  print(den._weights[0].new_array())

  den.update(1.0)
  print('New weights')
  print(den._weights[0].new_array())
  den.forward()
  print('Output, true value')
  print(den._outputs[0].new_array())
  print(dist._label_outputs[0].new_array())

if __name__ == '__main__':
  print('Running as main')
  dist = distributor(data = data, labels = labels, batch_size = 128, num_gpus = 1)
  dist.setup()
  outs = dist._outputs
  for layer in graph_network:
    layer.setup(outs)
    outs = layer._outputs
  soft = softmax_loss()
  soft.setup(outs)
  backs = dist._label_outputs
  soft.setup_backward(backs)
  backs = soft._backwards
  for layer in graph_network[::-1]:
    layer.setup_backward(backs)
    backs = layer._backwards 
  epoch = 0
  bm.startTiming('Main loop')
  while(epoch < 10):
    try:
      while(True):
        dist.forward()
        for layer in graph_network:
          layer.forward()
        soft.forward()
        soft.backward()
        for layer in graph_network[::-1]:
          layer.backward()
        for layer in graph_network:
          layer.update(0.01)
    except StopIteration:
      print('Finished epoch {:d}'.format(epoch))
      epoch += 1
      dist._cur_batch = 0
  rm.cuda.cuDeviceSynchronize()
  bm.endAllTiming()
  bm.getTimes()
  dist.forward()
  for layer in graph_network:
    layer.forward()
  print(np.argmax(graph_network[-1]._outputs[0].new_array(), axis = 1)) 
  print(np.argmax(dist._label_outputs[0].new_array(), axis = 1))
