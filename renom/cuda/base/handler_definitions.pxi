
_renom_cuda_handlers = {}

cdef createHandle():
  cdef RenomHandle *ret
  ret = <RenomHandle*> malloc(sizeof(RenomHandle));
  ret[0].stream = <cudaStream_t><uintptr_t> cuCreateStream()
  return <uintptr_t> &ret


@contextlib.contextmanager
def renom_handler():
  cdef renomHandle_t handle
  cdef int dev
  cudaGetDevice(&dev)

  if dev not in _renom_cuda_handlers:
    _renom_cuda_handlers[dev] = createHandle()

  yield _renom_cuda_handlers[dev]
