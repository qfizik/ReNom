import numpy as np
import renom as rm
from graph_element import graph_element
from graph_storage import GraphMultiStorage


class range_variable(graph_element):

    def __init__(self):
        super().__init__()

        self._num_gpus = 1
        self._gpus = [0]
        self._cur_num = 0
        self._shape = (1, 1)
        self._memory_info = GraphMultiStorage(
            shape=self._shape, gpus=self._num_gpus, allocate_backward=False)

    def forward(self):
        with rm.cuda.RenomHandler() as handle:
            arr = np.ones(self._shape).astype(rm.precision) * self._cur_num
            pin = handle.getPinnedMemory(arr)
            self._memory_info[0].to_gpu(pin)
        self._cur_num += 1
