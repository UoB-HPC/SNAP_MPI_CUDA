
#include "cuda_global.h"

void check_cuda_error(const char *msg, const int line, const char * file)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error in %s line %d: %s. %s\n", file, line, cudaGetErrorString(err), msg);
        exit(EXIT_FAILURE);
    }
} 

