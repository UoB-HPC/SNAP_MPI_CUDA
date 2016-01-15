
#define DENOMINATOR_INDEX(a,g,i,j,k,nang,ng,nx,ny) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(j))+((nang)*(ng)*(nx)*(ny)*(k)))
#define denominator(a,g,i,j,k) denominator[DENOMINATOR_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]

__global__ void calc_denominator(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,
    const double * __restrict__ mat_cross_section,
    const double * __restrict__ velocity_delta,
    const double * __restrict__ mu,
    const double * __restrict__ dd_i,
    const double * __restrict__ dd_j,
    const double * __restrict__ dd_k,
    double * __restrict__ denominator
    )
{
    size_t a = blockIdx.x * blockDim.x + threadIdx.x;
    size_t g = blockIdx.y * blockDim.y + threadIdx.y;

    if (a >= nang) return;
    if (g >= ng)   return;

    for (unsigned int k = 0; k < nz; k++)
        for (unsigned int j = 0; j < ny; j++)
            for (unsigned int i = 0; i < nx; i++)
                denominator(a,g,i,j,k) = 1.0 / (mat_cross_section[g] + velocity_delta[g] + mu[a]*dd_i[0] + dd_j[a] + dd_k[a]);
}
