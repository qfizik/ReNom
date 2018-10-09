from __future__ import print_function
import atexit
import sys
import traceback
import contextlib
import bisect
import threading

from libc.stdio cimport printf
cimport numpy as np
import numpy as pnp
cimport cython
from numbers import Number
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t, intptr_t
from libc.string cimport memcpy
from cuda_utils cimport _VoidPtr
import renom
from renom.config import precision
import collections

# Indicate Python started shutdown process
cdef int _python_shutdown = 0

@atexit.register
def on_exit():
    _python_shutdown = 1


@contextlib.contextmanager
def use_device(device_id):
    active = renom.cuda.is_cuda_active()
    cdef int cur

    if active:
        cur = cuGetDevice()
        cuSetDevice(device_id)  # switch dedice

    try:
        yield
    finally:
        if active:
            cuSetDevice(cur)   # restore device

def cuMalloc(uintptr_t nbytes):
    cdef void * p
    runtime_check(cudaMalloc( & p, nbytes))
    return < uintptr_t > p


cpdef cuMemset(uintptr_t ptr, int value, size_t size):
    p = <void * >ptr
    runtime_check(cudaMemset(p, value, size))
    return


'''
Creates a stream
The name is optional, if not given a default name will be chosen

A cudaStream_t type is a pointer to a CUstream_st struct
The function return value is the integer converted value of this pointer
To reuse this stream as a C-defined cudaStream_t variable, simply cast the
returned integer value back to cudaStream_t
'''
def cuCreateStream(name = None):
    cdef cudaStream_t stream
    cdef char* cname
    #runtime_check(cudaStreamCreateWithFlags( & stream, cudaStreamNonBlocking))
    runtime_check(cudaStreamCreate( & stream ))
    if name is not None:
      py_byte_string = name.encode("UTF-8")
      cname = py_byte_string
      nvtxNameCudaStreamA(stream, cname)
    return < uintptr_t > stream



cdef streamInsertEvent(cudaStream_t stream, cudaEvent_t event):
  runtime_check(cudaEventRecord(event, stream))


def heapReady(GPUHeap heap):
  ret = cudaEventQuery(heap.event)
  if ret == cudaSuccess:
    return True
  else:
    return False

def cuDestroyStream(uintptr_t stream):
    runtime_check(cudaStreamDestroy(<cudaStream_t> stream))

def cuResetDevice():
  runtime_check(cudaDeviceReset())

def cuGetMemInfo():
    cdef size_t free, total
    cudaMemGetInfo(&free, &total)
    return <long> free, <long> (total-free), <long> total # free, used, total


def cuSetDevice(int dev):
    runtime_check(cudaSetDevice(dev))


cpdef int cuGetDevice():
    cdef int dev
    runtime_check(cudaGetDevice(&dev))
    return dev

cpdef cuDeviceSynchronize():
    runtime_check(cudaDeviceSynchronize())


cpdef cuCreateCtx(device=0):
    cdef CUcontext ctx
    driver_check(cuCtxCreate( & ctx, 0, device))
    return int(ctx)


cpdef cuGetDeviceCxt():
    cdef CUdevice device
    driver_check(cuCtxGetDevice( & device))
    return int(device)


cpdef cuGetDeviceCount():
    cdef int count
    runtime_check(cudaGetDeviceCount( & count))
    return int(count)


cpdef cuGetDeviceProperty(device):
    cdef cudaDeviceProp property
    runtime_check(cudaGetDeviceProperties( & property, device))
    property_dict = {
        "name": property.name,
        "totalGlobalMem": property.totalGlobalMem,
        "sharedMemPerBlock": property.sharedMemPerBlock,
        "regsPerBlock": property.regsPerBlock,
        "warpSize": property.warpSize,
        "memPitch": property.memPitch,
        "maxThreadsPerBlock": property.maxThreadsPerBlock,
        "maxThreadsDim": property.maxThreadsDim,
        "maxGridSize": property.maxGridSize,
        "totalConstMem": property.totalConstMem,
        "major": property.major,
        "minor": property.minor,
        "clockRate": property.clockRate,
        "textureAlignment": property.textureAlignment,
        "deviceOverlap": property.deviceOverlap,
        "multiProcessorCount": property.multiProcessorCount,
        "kernelExecTimeoutEnabled": property.kernelExecTimeoutEnabled,
        "computeMode": property.computeMode,
        "concurrentKernels": property.concurrentKernels,
        "ECCEnabled": property.ECCEnabled,
        "pciBusID": property.pciBusID,
        "pciDeviceID": property.pciDeviceID,
        "tccDriver": property.tccDriver,
    }

    return property_dict


cpdef cuFree(uintptr_t ptr):
    p = <void * >ptr
    runtime_check(cudaFree(p))
    return

# cuda runtime check
cpdef runtime_check(error):
    if error != cudaSuccess:
        error_msg = cudaGetErrorString(error)
        raise Exception("CUDA Error: #{}|||{}".format(error,error_msg))
    return

# cuda runtime check
cpdef driver_check(error):
    cdef char * string
    if error != 0:
        cuGetErrorString(error, < const char**> & string)
        error_msg = str(string)
        raise Exception(error_msg)
    return

# Memcpy
# TODO: in memcpy function, dest arguments MUST come first!

cdef void cuMemcpyH2D(void* cpu_ptr, uintptr_t gpu_ptr, int size):
    # cpu to gpu
    runtime_check(cudaMemcpy(<void *>gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice))
    return

def queryDeviceProperties():
    cdef cudaDeviceProp props
    cdef int device = 0
    cudaGetDeviceProperties(&props,device)
    print("Device name is {}".format(
      props.name
    ))
    print("Device has compute capability {:d}.{:d}".format(
      props.major, props.minor
    ))
    print("Device has {:d} engines".format(
      props.asyncEngineCount
    ))


cdef cuMemcpyD2H(uintptr_t gpu_ptr, void *cpu_ptr, int size):
    # gpu to cpu
    runtime_check(cudaMemcpy(cpu_ptr, <void *>gpu_ptr, size, cudaMemcpyDeviceToHost))
    return

cdef cuMemcpyD2Hvar(uintptr_t gpu_ptr, void *cpu_ptr, int size, uintptr_t stream):
    # gpu to cpu
    runtime_check(cudaMemcpy(cpu_ptr, <void *>gpu_ptr, size, cudaMemcpyDeviceToHost))
    return


def cuMemcpyD2D(uintptr_t gpu_ptr1, uintptr_t gpu_ptr2, int size):
    # gpu to gpu
    runtime_check(cudaMemcpy(< void*>gpu_ptr2, < const void*>gpu_ptr1, size, cudaMemcpyDeviceToDevice))
    return

cdef void cuMemcpyH2DAsync(void* cpu_ptr, uintptr_t gpu_ptr, int size, uintptr_t stream):
    # cpu to gpu
    runtime_check(cudaMemcpyAsync(<void *>gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice, <cudaStream_t> stream))
    return



def cuMemcpyD2HAsync(uintptr_t gpu_ptr, np.ndarray[float, ndim=1, mode="c"] cpu_ptr, int size, int stream=0):
    # gpu to cpu
    runtime_check(cudaMemcpyAsync( & cpu_ptr[0], < const void*>gpu_ptr, size, cudaMemcpyDeviceToHost, < cudaStream_t > stream))
    return


cpdef cuMemcpyD2DAsync(uintptr_t gpu_ptr1, uintptr_t gpu_ptr2, int size, int stream=0):
    # gpu to gpu
    runtime_check(cudaMemcpyAsync( < void*>gpu_ptr2, < const void*>gpu_ptr1, size, cudaMemcpyDeviceToDevice, < cudaStream_t > stream))
    return

def check_heap_device(*heaps):
    devices = {h._ptr.device_id for h in heaps if isinstance(h, renom.core.GPUValue)}

    current = {cuGetDevice()}
    if devices != current:
        raise RuntimeError('Invalid device_id: %s currennt: %s' % (devices, current))


cdef class PinnedMemory(object):
    def __init__(self, array_to_pin, stream):
        cdef int i
        self.size = array_to_pin.itemsize*array_to_pin.size
        self.shape = array_to_pin.shape
        self.dtype = array_to_pin.dtype
        self.stream = <cudaStream_t><uintptr_t> stream
        runtime_check(cudaEventCreate(&self.event))
        runtime_check(cudaMallocHost(&self.memory_ptr, self.size))

    def __dealloc__(self):
        runtime_check(cudaFreeHost(self.memory_ptr))

    def get_size(self):
        return <uintptr_t> self.size

    def pin(self, np.ndarray arr):
        cdef size_t sz = arr.descr.itemsize * arr.size
        if sz != self.size:
          raise ValueError("Trying to pin array of different size than originally initialized")
        memcpy(self.memory_ptr, <void*> arr.data, self.size)

    def __int__(self):
        return <uintptr_t> self.memory_ptr

    def get_ptr(self):
        return <uintptr_t> self.memory_ptr

    @property
    def dtype(self):
        return self.dtype

    @property
    def shape(self):
        return self.shape

    @property
    def nbytes(self):
        return self.size

cdef void *get_pointer(pyobject):
  cdef void* ptr
  cdef _VoidPtr vptr
  if isinstance(pyobject, pnp.ndarray):
    buf = pyobject.ravel()
    vptr = _VoidPtr(buf)
    ptr = <void*> vptr.ptr
  elif isinstance(pyobject, PinnedMemory):
    ptr = <void*><uintptr_t> pyobject.get_ptr()
  else:
    raise TypeError("Unknown cpu ptr type, expected ndarray or pinned memory")
  return ptr

cdef class GPUHeap(object):
    def __init__(self, nbytes, ptr, device_id, stream=0):
        self.ptr = ptr
        self.nbytes = nbytes
        self.device_id = device_id
        # The GPUHeap sets its refcount to 0, as it does not personally know if it is
        # to be owned during creation. Refcount is instead managed in GPUValue.
        self.refcount = 0
        # The stream is decided by the allocator and given to all subsequently
        # constructed GPUHeaps. All Memcpy operations will occur on the same
        # stream.
        self._mystream = stream
        cudaEventCreate(&self.event)

    def __int__(self):
        return self.ptr

    def __dealloc__(self):
        # Python functions should be avoided as far as we can

        cdef int cur
        cdef cudaError_t err
        cdef const char *errstr;

        cudaGetDevice(&cur)

        try:
            err = cudaSetDevice(self.device_id)
            if err == cudaSuccess:
                err = cudaFree(<void * >self.ptr)

            if err != cudaSuccess:
                errstr = cudaGetErrorString(err)
                if _python_shutdown == 0:
                    s =  errstr.decode('utf-8', 'replace')
                    print("Error in GPUHeap.__dealloc__():", err, s, file=sys.stderr)
                else:
                    printf("Error in GPUHeap.__dealloc__(): %s\n", errstr)

        finally:
            cudaSetDevice(cur)


    cpdef memcpyH2D(self, pyobject_to_load, size_t nbytes):
        # todo: this copy is not necessary
        # This pointer is already exposed with the Cython numpy implementation

        if isinstance(pyobject_to_load, pnp.ndarray):
            self.loadNumpy(pyobject_to_load, nbytes)
        elif isinstance(pyobject_to_load, PinnedMemory):
            self.loadPinned(pyobject_to_load, nbytes)

    cdef loadNumpy(self, numpy_to_load, size_t nbytes):
      cdef void* ptr
      cdef _VoidPtr vptr
      buf = numpy_to_load.ravel()
      vptr = _VoidPtr(buf)
      ptr = <void*> vptr.ptr
      cuMemcpyH2D(ptr, self.ptr, nbytes)

    cdef loadPinned(self, PinnedMemory pinned_to_load, size_t nbytes):
      cdef void* ptr
      ptr = <void*><uintptr_t> pinned_to_load
      cuMemcpyH2DAsync(ptr, self.ptr, nbytes, <uintptr_t> pinned_to_load.stream)
      streamInsertEvent(<cudaStream_t><uintptr_t> pinned_to_load.stream, <cudaEvent_t><uintptr_t> pinned_to_load.event)


    cpdef memcpyD2H(self, cpu_ptr, size_t nbytes):
        shape = cpu_ptr.shape
        cpu_ptr = pnp.reshape(cpu_ptr, -1)

        cdef _VoidPtr ptr = _VoidPtr(cpu_ptr)

        with renom.cuda.use_device(self.device_id):
            cuMemcpyD2H(self.ptr, ptr.ptr, nbytes)

        pnp.reshape(cpu_ptr, shape)

    cpdef memcpyD2D(self, gpu_ptr, size_t nbytes):
        assert self.device_id == gpu_ptr.device_id
        with renom.cuda.use_device(self.device_id):
            cuMemcpyD2D(self.ptr, gpu_ptr.ptr, nbytes)

    cpdef copy_from(self, other, size_t nbytes):
        cdef void *buf
        cdef int ret
        cdef uintptr_t src, dest

        assert nbytes <= self.nbytes
        assert nbytes <= other.nbytes

        n = min(self.nbytes, other.nbytes)
        if self.device_id == other.device_id:
            # self.memcpyD2D(other, n)
            other.memcpyD2D(self, n)
        else:
            runtime_check(cudaDeviceCanAccessPeer(&ret, self.device_id, other.device_id))
            if ret:
                src = other.ptr
                dest = self.ptr
                runtime_check(cudaMemcpyPeer(<void *>dest, self.device_id, <void*>src, other.device_id, nbytes))
            else:
                buf = malloc(n)
                if not buf:
                    raise MemoryError()
                try:
                    with renom.cuda.use_device(other.device_id):
                        cuMemcpyD2H(other.ptr, buf, n)

                    with renom.cuda.use_device(self.device_id):
                        cuMemcpyH2D(buf, self.ptr, n)

                finally:
                    free(buf)


cdef class GpuAllocator(object):

    def __init__(self):
        self._pool_lists = collections.defaultdict(list)
        # We create one stream for all the GPUHeaps to share
        self._rlock = threading.RLock()

    @property
    def pool_list(self):
        device = cuGetDevice()
        return self._pool_lists[device]

    cpdef GPUHeap malloc(self, size_t nbytes):
        cdef GPUHeap pool = self.getAvailablePool(nbytes)
        if pool is None:
            ptr = cuMalloc(nbytes)
            pool = GPUHeap(nbytes=nbytes, ptr=ptr, device_id=cuGetDevice())
        return pool


    cpdef free(self, GPUHeap pool):
        '''
        When a pool is to be freed, we first record the current status of the stream in which it was used,
        so as to make sure that it is not prematurely released for use by other GPUValues requesting a pool.
        '''
        if _python_shutdown:
            return


        if pool.nbytes:
            device_id = pool.device_id

            with self._rlock:
                self._pool_lists[device_id]
                index = bisect.bisect(self._pool_lists[device_id], (pool.nbytes,))
                self._pool_lists[device_id].insert(index, (pool.nbytes, pool))

    cpdef GPUHeap getAvailablePool(self, size_t size):
        pool = None
        '''
        We will be looking through the currently available pools and we demand that they
        big enough to fit all our requested data, but we allow for pools that are slightly
        larger than what is requested
        '''
        cdef size_t min_requested = size
        cdef size_t max_requested = size * 2 + 4096
        cdef size_t idx, i
        cdef GPUHeap p

        with self._rlock:
            device = cuGetDevice()
            pools = self._pool_lists[device]

            idx = bisect.bisect_left(pools, (size,))


            for i in range(idx, len(pools)):
                _, p = pools[i]
                if p.nbytes >= max_requested:
                    break

                if min_requested <= p.nbytes and heapReady(p):
                    pool = p
                    del pools[i]
                    break

        return pool

    cpdef release_pool(self, deviceID=None):
        if deviceID is None:
            self._pool_lists = collections.defaultdict(list)
        else:
            del self._pool_lists[deviceID]


gpu_allocator = GpuAllocator()

cdef GpuAllocator c_gpu_allocator
c_gpu_allocator = gpu_allocator

cpdef GpuAllocator get_gpu_allocator():
    return c_gpu_allocator


cpdef _cuSetLimit(limit, value):
    cdef size_t c_value=999;

    cuInit(0)

    ret = cuCtxGetLimit(&c_value, limit)

    cuCtxSetLimit(limit, value)
