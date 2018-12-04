import renom as rm
import numpy as np

def compare(nd_value, ad_value):
  print('ad=')
  print(ad_value)
  print('nd=')
  print(nd_value)
  assert np.allclose(nd_value, ad_value, atol = 1e-5, rtol = 1e-3)


def test_basic_add():

  v1 = np.random.rand(2,2)
  v2 = np.random.rand(2,2)
  v3 = v1 + v2
  v4 = np.random.rand(2,2)
  v5 = v3 + v4
  
  g1 = rm.graph.StaticVariable(v1)
  g2 = rm.graph.StaticVariable(v2)
  g3 = g1 + g2
  g4 = rm.graph.StaticVariable(v4)
  g5 = g3 + g4
  
  compare(v5, g5.as_ndarray())
  
  new_v1 = np.random.rand(2,2)
  g1.value = new_v1

  new_v5 = new_v1 + v2 + v4
  g5.forward()
  compare(new_v5, g5.as_ndarray()) 
