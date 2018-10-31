from libc.stdlib cimport *
from libc.stdint cimport uintptr_t
from renom.cuda.base.cuda_base cimport *

cdef extern from "nccl.h":
  ctypedef struct ncclComm
  ctypedef ncclComm* ncclComm_t
  ctypedef struct ncclUniqueId

  ctypedef enum ncclDataType_t:
    ncclChar,
    ncclUint8,
    ncclInt,
    ncclUint32,
    ncclInt64,
    ncclUint64,
    ncclHalf,
    ncclFloat,
    Double

  ctypedef enum ncclRedOp_t:
    ncclSum,
    ncclProd,
    ncclMin,
    ncclMax

  ctypedef enum ncclResult_t:
    ncclSuccess,
    ncclUnhandledCudaError,
    ncclSystemError,
    ncclInternalError,
    ncclInvalidArgument,
    ncclInvalidUsage

  ncclResult_t ncclGetVersion(int *version)
  ncclResult_t ncclCommInitAll(ncclComm_t *comms, int nranks, int *devs);
  ncclResult_t ncclCommDestroy(ncclComm_t comm)
  char *ncclGetErrorString(ncclResult_t)

  ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t elements, ncclDataType_t elementtype, ncclRedOp_t redop, ncclComm_t comm, cudaStream_t stream)

  ncclResult_t ncclGroupStart()
  ncclResult_t ncclGroupEnd()
