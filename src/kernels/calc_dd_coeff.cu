
__global__
void calc_dd_coeff(
    const double dx,
    const double dy,
    const double dz,
    const double * __restrict__ eta,
    const double * __restrict__ xi,
    double * __restrict__ dd_i,
    double * __restrict__ dd_j,
    double * __restrict__ dd_k
    )
{
    size_t a = blockIdx.x * blockDim.x + threadIdx.x;

    // There is only one dd_i so just get the first thread to do this
    if (a == 0 && blockIdx.x == 0)
        dd_i[0] = 2.0 / dx;

    dd_j[a] = (2.0 / dy) * eta[a];
    dd_k[a] = (2.0 / dz) * xi[a];
}
