
#include "cuda_buffers.h"

void allocate_buffers(
    struct problem * problem, struct rankinfo * rankinfo,
    struct buffers * buffers)
{

    // Angular flux
    for (int i = 0; i < 8; i++)
    {
        cudaMalloc(
            &(buffers->angular_flux_in[i]),
            sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz
        );
        check_cuda("Creating an angular flux in buffer");

        cudaMalloc(
            &(buffers->angular_flux_out[i]),
            sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz
        );
        check_cuda("Creating an angular flux out buffer");
    }

    // Edge fluxes
    cudaMalloc(
         &buffers->flux_i,
        sizeof(double)*problem->nang*problem->ng*rankinfo->ny*rankinfo->nz
    );
    check_cuda("Creating flux_i buffer");

    cudaMalloc(
        &buffers->flux_j,
        sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->nz
    );
    check_cuda("Creating flux_j buffer");

    cudaMalloc(
        &buffers->flux_k,
        sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny
    );
    check_cuda("Creating flux_k buffer");

    // Scalar flux
    cudaMalloc(
        &buffers->scalar_flux, 
        sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz
    );
    check_cuda("Creating scalar flux buffer");


    if (problem->cmom-1 > 0)
    {
        cudaMalloc(
            &buffers->scalar_flux_moments,
            sizeof(double)*(problem->cmom-1)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz
        );
        check_cuda("Creating scalar flux moments buffer");
    }
    else
    {
        buffers->scalar_flux_moments = NULL;
    }

    // Weights and cosines
    cudaMalloc(&buffers->quad_weights, sizeof(double)*problem->nang);
    check_cuda("Creating quadrature weights buffer");
    cudaMalloc(&buffers->mu, sizeof(double)*problem->nang);
    check_cuda("Creating mu cosine buffer");
    cudaMalloc(&buffers->eta, sizeof(double)*problem->nang);
    check_cuda("Creating eta cosine buffer");
    cudaMalloc(&buffers->xi, sizeof(double)*problem->nang);
    check_cuda("Creating xi cosine buffer");

    // Scattering coefficient
    cudaMalloc(&buffers->scat_coeff,
        sizeof(double)*problem->nang*problem->cmom*8);
    check_cuda("Creating scattering coefficient buffer");

    // Material cross section
    cudaMalloc(&buffers->mat_cross_section, sizeof(double)*problem->ng);
    check_cuda("Creating material cross section buffer");

    // Source terms
    cudaMalloc(&buffers->fixed_source,
        sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz);
    check_cuda("Creating fixed source buffer");
    cudaMalloc(&buffers->outer_source,
        sizeof(double)*problem->cmom*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz);
    check_cuda("Creating outer source buffer");
    cudaMalloc(&buffers->inner_source,
        sizeof(double)*problem->cmom*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz);
    check_cuda("Creating inner source buffer");

    // Scattering terms
    cudaMalloc(&buffers->scattering_matrix,
        sizeof(double)*problem->nmom*problem->ng*problem->ng);
    check_cuda("Creating scattering matrix buffer");

    // Diamond diference co-efficients
    cudaMalloc(&buffers->dd_i, sizeof(double));
    check_cuda("Creating i diamond difference coefficient");
    cudaMalloc(&buffers->dd_j, sizeof(double)*problem->nang);
    check_cuda("Creating j diamond difference coefficient");
    cudaMalloc(&buffers->dd_k, sizeof(double)*problem->nang);
    check_cuda("Creating k diamond difference coefficient");

    // Velocities
    cudaMalloc(&buffers->velocities, sizeof(double)*problem->ng);
    check_cuda("Creating velocity buffer");
    cudaMalloc(&buffers->velocity_delta, sizeof(double)*problem->ng);
    check_cuda("Creating velocity delta buffer");

    // Denominator array
    cudaMalloc(&buffers->denominator,
        sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz);
    check_cuda("Creating denominator buffer");

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

