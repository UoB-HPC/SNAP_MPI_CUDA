
#include "source.h"


void compute_outer_source(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    struct buffers * buffers
    )
{
    dim3 grid(rankinfo->nx, rankinfo->ny, rankinfo->nz);
    dim3 threads(1,1,1);
    calc_outer_source<<< grid, threads >>>(
        rankinfo->nx, rankinfo->ny, rankinfo->nz,
        problem->ng, problem->cmom, problem->nmom,
        buffers->fixed_source, buffers->scattering_matrix,
        buffers->scalar_flux, buffers->scalar_flux_moments,
        buffers->outer_source
    );
    check_cuda("Enqueue outer source kernel");
}


void compute_inner_source(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    struct buffers * buffers
    )
{
    dim3 grid(rankinfo->nx, rankinfo->ny, rankinfo->nz);
    dim3 threads(1, 1, 1);
    calc_inner_source<<< grid, threads >>>(
        rankinfo->nx, rankinfo->ny, rankinfo->nz,
        problem->ng, problem->cmom, problem->nmom,
        buffers->outer_source, buffers->scattering_matrix,
        buffers->scalar_flux, buffers->scalar_flux_moments,
        buffers->inner_source
    );
    check_cuda("Enqueue inner source kernel");
}

