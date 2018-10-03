cdef extern from "renomhandle.h":
  ctypedef struct RenomHandle:
    char *name
    cudaStream_t stream

  ctypedef RenomHandle* renomHandle_t
