#ifndef __CUDA_HANDLE_H__
#define __CUDA_HANDLE_H__

#include "cuda_runtime.h"

typedef struct _Renom_Handle
{
  char *name;
  cudaStream_t stream;
} RenomHandle;

typedef RenomHandle* renomHandle_t;

#endif
