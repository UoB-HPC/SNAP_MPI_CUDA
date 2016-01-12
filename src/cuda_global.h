
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_kernels.h"

#ifndef BLOCK_SIZE_2D
// So blocks are 256 = 16*16 threads
#define BLOCK_SIZE_2D 16
#endif

/** \file
* \brief Basic CUDA routines and global structures
*/

#ifdef __cplusplus
extern "C"
{
#endif

/**
@{
\brief Check for CUDA errors */
void check_cuda_error(const char *msg, const int line, const char * file);
#define check_cuda(msg) check_cuda_error((msg),__LINE__,__FILE__)
/** @}*/

#ifdef __cplusplus
}
#endif

