
// Calculate the time absorbtion coefficient
__global__ void calc_velocity_delta(
    const double * restrict velocities,
    const double dt,
    double * restrict velocity_delta
    )
{
    size_t g = blockIdx.x * blockDim.x + threadIdx.x;
    velocity_delta[g] = 2.0 / (dt * velocities[g]);

}
