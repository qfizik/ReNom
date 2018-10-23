import renom as rm
import numpy as np
from static_variable import static_variable
from range_variable import range_variable
rm.cuda.set_cuda_active()

a = np.ones((1, 1)).astype(rm.precision)

a = static_variable(a)
b = range_variable()

c = a + b
print(c)

for _ in range(10):
  print(c)
