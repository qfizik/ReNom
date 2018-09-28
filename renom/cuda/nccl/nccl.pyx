
def ncclCheck(err):
  if err != ncclSuccess:
    print("Error in NCCL: {}", err)
    print("{}",ncclGetErrorString(err))

def cuGroupStart(): 
  ncclCheck(ncclGroupStart())

def cuGroupEnd():
  ncclCheck(ncclGroupEnd())

cdef class DeviceCommunicator:

  cdef int ndev
  cdef int *devs
  cdef ncclComm_t *comms

  def __cinit__(self, int num_device): 
    cdef int i
    self.ndev = <int>num_device
    self.devs = <int*>malloc(self.ndev * sizeof(int))
    for i in range(self.ndev):
      self.devs[i] = i
    self.comms = <ncclComm_t*>malloc(self.ndev * sizeof(ncclComm_t))
    ncclCheck(ncclCommInitAll(self.comms, self.ndev, self.devs))

  def __dealloc__(self):
    for i in range(self.ndev):
      ncclCheck(ncclCommDestroy(self.comms[i]))
    free(self.devs)
    free(self.comms)
