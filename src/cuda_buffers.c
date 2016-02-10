
#include "cuda_buffers.h"

void allocate_buffers(
    struct problem * problem, struct rankinfo * rankinfo,
    struct buffers * buffers)
{

    // Angular flux
    for (int i = 0; i < 8; i++)
    {
        cudaMalloc(
            (void **)&(buffers->angular_flux_in[i]),
            sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz
        );
        check_cuda("Creating an angular flux in buffer");

        cudaMalloc(
            (void **)&(buffers->angular_flux_out[i]),
            sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz
        );
        check_cuda("Creating an angular flux out buffer");
    }

    // Edge fluxes
    cudaMalloc(
         (void **)&buffers->flux_i,
        sizeof(double)*problem->nang*problem->ng*rankinfo->ny*rankinfo->nz
    );
    check_cuda("Creating flux_i buffer");

    cudaMalloc(
        (void **)&buffers->flux_j,
        sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->nz
    );
    check_cuda("Creating flux_j buffer");

    cudaMalloc(
        (void **)&buffers->flux_k,
        sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny
    );
    check_cuda("Creating flux_k buffer");

    // Scalar flux
    cudaMalloc(
        (void **)&buffers->scalar_flux,
        sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz
    );
    check_cuda("Creating scalar flux buffer");


    if (problem->cmom-1 > 0)
    {
        cudaMalloc(
            (void **)&buffers->scalar_flux_moments,
            sizeof(double)*(problem->cmom-1)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz
        );
        check_cuda("Creating scalar flux moments buffer");
    }
    else
    {
        buffers->scalar_flux_moments = NULL;
    }

    // Weights and cosines
    cudaMalloc((void **)&buffers->quad_weights, sizeof(double)*problem->nang);
    check_cuda("Creating quadrature weights buffer");
    cudaMalloc((void **)&buffers->mu, sizeof(double)*problem->nang);
    check_cuda("Creating mu cosine buffer");
    cudaMalloc((void **)&buffers->eta, sizeof(double)*problem->nang);
    check_cuda("Creating eta cosine buffer");
    cudaMalloc((void **)&buffers->xi, sizeof(double)*problem->nang);
    check_cuda("Creating xi cosine buffer");

    // Scattering coefficient
    cudaMalloc((void **)&buffers->scat_coeff,
        sizeof(double)*problem->nang*problem->cmom*8);
    check_cuda("Creating scattering coefficient buffer");

    // Material cross section
    cudaMalloc((void **)&buffers->mat_cross_section, sizeof(double)*problem->ng);
    check_cuda("Creating material cross section buffer");

    // Source terms
    cudaMalloc((void **)&buffers->fixed_source,
        sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz);
    check_cuda("Creating fixed source buffer");
    cudaMalloc((void **)&buffers->outer_source,
        sizeof(double)*problem->cmom*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz);
    check_cuda("Creating outer source buffer");
    cudaMalloc((void **)&buffers->inner_source,
        sizeof(double)*problem->cmom*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz);
    check_cuda("Creating inner source buffer");

    // Scattering terms
    cudaMalloc((void **)&buffers->scattering_matrix,
        sizeof(double)*problem->nmom*problem->ng*problem->ng);
    check_cuda("Creating scattering matrix buffer");

    // Diamond diference co-efficients
    cudaMalloc((void **)&buffers->dd_i, sizeof(double));
    check_cuda("Creating i diamond difference coefficient");
    cudaMalloc((void **)&buffers->dd_j, sizeof(double)*problem->nang);
    check_cuda("Creating j diamond difference coefficient");
    cudaMalloc((void **)&buffers->dd_k, sizeof(double)*problem->nang);
    check_cuda("Creating k diamond difference coefficient");

    // Velocities
    cudaMalloc((void **)&buffers->velocities, sizeof(double)*problem->ng);
    check_cuda("Creating velocity buffer");
    cudaMalloc((void **)&buffers->velocity_delta, sizeof(double)*problem->ng);
    check_cuda("Creating velocity delta buffer");
}

void swap_angular_flux_buffers(struct buffers * buffers)
{
    for (int i = 0; i < 8; i++)
    {
        double *tmp;
        tmp = buffers->angular_flux_in[i];
        buffers->angular_flux_in[i] = buffers->angular_flux_out[i];
        buffers->angular_flux_out[i] = tmp;
    }
}

