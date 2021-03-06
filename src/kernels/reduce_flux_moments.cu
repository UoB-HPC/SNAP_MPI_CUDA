
#define ANGULAR_FLUX_INDEX(a,g,i,j,k,nang,ng,nx,ny) ((a)+((nang)*(g))+((nang)*(ng)*(i))+((nang)*(ng)*(nx)*(j))+((nang)*(ng)*(nx)*(ny)*(k)))
#define SCALAR_FLUX_MOMENTS_INDEX(m,g,i,j,k,cmom,ng,nx,ny) ((m)+((cmom-1)*(g))+((cmom-1)*(ng)*(i))+((cmom-1)*(ng)*(nx)*(j))+((cmom-1)*(ng)*(nx)*(ny)*(k)))
#define SCAT_COEFF_INDEX(a,l,o,nang,cmom) ((a)+((nang)*(l))+((nang)*(cmom)*o))

#define angular_flux_in_0(a,g,i,j,k) angular_flux_in_0[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_1(a,g,i,j,k) angular_flux_in_1[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_2(a,g,i,j,k) angular_flux_in_2[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_3(a,g,i,j,k) angular_flux_in_3[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_4(a,g,i,j,k) angular_flux_in_4[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_5(a,g,i,j,k) angular_flux_in_5[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_6(a,g,i,j,k) angular_flux_in_6[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_in_7(a,g,i,j,k) angular_flux_in_7[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_0(a,g,i,j,k) angular_flux_out_0[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_1(a,g,i,j,k) angular_flux_out_1[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_2(a,g,i,j,k) angular_flux_out_2[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_3(a,g,i,j,k) angular_flux_out_3[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_4(a,g,i,j,k) angular_flux_out_4[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_5(a,g,i,j,k) angular_flux_out_5[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_6(a,g,i,j,k) angular_flux_out_6[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]
#define angular_flux_out_7(a,g,i,j,k) angular_flux_out_7[ANGULAR_FLUX_INDEX((a),(g),(i),(j),(k),nang,ng,nx,ny)]

#define scalar_flux_moments(l,g,i,j,k) scalar_flux_moments[SCALAR_FLUX_MOMENTS_INDEX((l),(g),(i),(j),(k),cmom,ng,nx,ny)]
#define scat_coeff(a,l,o) scat_coeff[SCAT_COEFF_INDEX((a),(l),(o),nang,cmom)]


// We want to perform a weighted sum of angles in each cell in each energy group for each moment
// One work-group per cell per energy group, and reduce within a work-group
// Work-groups must be power of two sized
__global__ void reduce_flux_moments(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,
    const unsigned int cmom,

    const double * __restrict__ angular_flux_in_0,
    const double * __restrict__ angular_flux_in_1,
    const double * __restrict__ angular_flux_in_2,
    const double * __restrict__ angular_flux_in_3,
    const double * __restrict__ angular_flux_in_4,
    const double * __restrict__ angular_flux_in_5,
    const double * __restrict__ angular_flux_in_6,
    const double * __restrict__ angular_flux_in_7,

    const double * __restrict__ angular_flux_out_0,
    const double * __restrict__ angular_flux_out_1,
    const double * __restrict__ angular_flux_out_2,
    const double * __restrict__ angular_flux_out_3,
    const double * __restrict__ angular_flux_out_4,
    const double * __restrict__ angular_flux_out_5,
    const double * __restrict__ angular_flux_out_6,
    const double * __restrict__ angular_flux_out_7,

    const double * __restrict__ velocity_delta,
    const double * __restrict__ quad_weights,
    const double * __restrict__ scat_coeff,

    double * __restrict__ scalar_flux_moments
    )
{
    extern __shared__ double local_scalar[];

    const size_t a = threadIdx.x;
    const size_t g = blockIdx.x;

    const size_t global_id = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t i = global_id % nx;
    const size_t j = (global_id / nx) % ny;
    const size_t k = global_id / (nx * ny);

    if (global_id >= nx * ny * nz) return;

    for (unsigned int l = 0; l < cmom-1; l++)
    {
        // Load into local memory
        local_scalar[a + (threadIdx.y * blockDim.x)] = 0.0;
        for (unsigned int aa = a; aa < nang; aa += blockDim.x)
        {
            const double w = quad_weights[aa];
            if (velocity_delta[g] != 0.0)
            {
                local_scalar[a + (threadIdx.y * blockDim.x)] +=
                    scat_coeff(aa,l+1,0) * w * (0.5 * (angular_flux_out_0(aa,g,i,j,k) + angular_flux_in_0(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,1) * w * (0.5 * (angular_flux_out_1(aa,g,i,j,k) + angular_flux_in_1(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,2) * w * (0.5 * (angular_flux_out_2(aa,g,i,j,k) + angular_flux_in_2(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,3) * w * (0.5 * (angular_flux_out_3(aa,g,i,j,k) + angular_flux_in_3(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,4) * w * (0.5 * (angular_flux_out_4(aa,g,i,j,k) + angular_flux_in_4(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,5) * w * (0.5 * (angular_flux_out_5(aa,g,i,j,k) + angular_flux_in_5(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,6) * w * (0.5 * (angular_flux_out_6(aa,g,i,j,k) + angular_flux_in_6(aa,g,i,j,k))) +
                    scat_coeff(aa,l+1,7) * w * (0.5 * (angular_flux_out_7(aa,g,i,j,k) + angular_flux_in_7(aa,g,i,j,k)));
            }
            else
            {
                local_scalar[a + (threadIdx.y * blockDim.x)] +=
                    scat_coeff(aa,l+1,0) * w * angular_flux_out_0(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,1) * w * angular_flux_out_1(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,2) * w * angular_flux_out_2(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,3) * w * angular_flux_out_3(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,4) * w * angular_flux_out_4(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,5) * w * angular_flux_out_5(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,6) * w * angular_flux_out_6(aa,g,i,j,k) +
                    scat_coeff(aa,l+1,7) * w * angular_flux_out_7(aa,g,i,j,k);
            }
        }

        __syncthreads();

        // Reduce in local memory
        for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2)
        {
            if (a < offset)
            {
                local_scalar[a + (threadIdx.y * blockDim.x)] += local_scalar[a + offset + (threadIdx.y * blockDim.x)];
            }
            __syncthreads();
        }

        // Save result
        if (a == 0)
        {
            scalar_flux_moments(l,g,i,j,k) = local_scalar[0 + (threadIdx.y * blockDim.x)];
        }
    } // End of moment loop
}
