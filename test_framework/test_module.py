from graph_element import *
from static_variable import *
from range_variable import *
from distributor import *
from dense_graph_element import *
from mean_squared_error import *
from add import *
import h5py
import numpy as np
import renom as rm
rm.set_cuda_active()



if __name__ == '__main__':
  

  with h5py.File('mnist-dataset.hdf5','r') as f:
    imgs = np.array(f['training-set']['Images']).astype(rm.precision).reshape(-1, 28 * 28)
    lbls = np.array(f['training-set']['Labels']).astype(rm.precision).reshape(-1, 1)

  print ('Starting Test')
  a = np.ones((1,2)) * 2
  b = np.ones((1,1)) * 3

  Dist = distributor(imgs, lbls, batch_size = 512)

  a = static_variable(a)
  b = static_variable(b)


  D1 = dense_graph_element(output_size = 1000)
  D2 = dense_graph_element(output_size = 500)
  D3 = dense_graph_element(output_size = 1)
  L = mean_squared_element()

  D1.connect(Dist)

  D2.connect(D1)

  D3.connect(D2)

  L.connect(previous_element = D3, labels = Dist)

  #L.print_tree()
  for epoch in range(10):
    try:
      while(True):
        L.update()
    except StopIteration:
      Dist.reset()

  L.forward()
  print(D3)
