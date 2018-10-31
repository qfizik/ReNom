
import renom as rm

cpdef ncclCheck(err):
  if err != ncclSuccess:
    print("Error in NCCL: {}", err)
    print("{}",ncclGetErrorString(err))

cpdef cuGroupStart():
  ncclCheck(ncclGroupStart())

cpdef cuGroupEnd():
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

  def allReduce(self, gpuvarray):
      cdef int i
      cdef size_t elems = 1
      for i in range(len(gpuvarray[0].shape)):
        elems *= gpuvarray[0].shape[i]
      ncclCheck(ncclGroupStart())
      for i in range(self.ndev):
        with rm.cuda.RenomHandler(i) as handle:
          ncclCheck(ncclAllReduce(
              <const void*><uintptr_t>gpuvarray[i]._ptr,
              <void*><uintptr_t>gpuvarray[i]._ptr,
              elems,
              ncclFloat,
              ncclSum,
              self.comms[i],
              <cudaStream_t><uintptr_t> handle.stream
          ))
      ncclCheck(ncclGroupEnd())

