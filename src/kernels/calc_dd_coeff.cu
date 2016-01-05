
__global__ void calc_dd_coeff(
    const double dx,
    const double dy,
    const double dz,
    const double * restrict eta,
    const double * restrict xi,
    double * restrict dd_i,
    double * restrict dd_j,
    double * restrict dd_k
    )
{
    size_t a = blockIdx.x * blockDim.x + threadIdx.x;

    // There is only one dd_i so just get the first thread to do this
    if (a == 0 && blockIdx.x == 0)
        dd_i[0] = 2.0 / dx;

    dd_j[a] = (2.0 / dy) * eta[a];
    dd_k[a] = (2.0 / dz) * xi[a];
}
