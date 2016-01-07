
#include "cuda_buffers.h"


void check_device_memory_requirements(
    struct problem * problem, struct rankinfo * rankinfo,
    struct context * context)
{
    cl_int err;
    cl_ulong global;
    err = clGetDeviceInfo(context->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global, NULL);
    check_ocl(err, "Getting device memory size");

    cl_ulong total = 0;
    // Add up the memory requirements, in bytes.
    total += problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz*8;
    total += problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz*8;
    total += problem->nang*problem->ng*rankinfo->ny*rankinfo->nz;
    total += problem->nang*problem->ng*rankinfo->nx*rankinfo->nz;
    total += problem->nang*problem->ng*rankinfo->nx*rankinfo->ny;
    total += problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    if (problem->cmom-1 == 0)
        total += problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    else
        total += 1;
    total += problem->nang;
    total += problem->nang;
    total += problem->nang;
    total += problem->nang;
    total += problem->nang*problem->cmom*8;
    total += problem->ng;
    total += problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    total += problem->cmom*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    total += problem->cmom*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    total += problem->nmom*problem->ng*problem->ng;
    total += 1;
    total += problem->nang;
    total += problem->nang;
    total += problem->ng;
    total += problem->ng;
    total += problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    total *= sizeof(double);

    if (global < total)
    {
        fprintf(stderr,"Error: Device does not have enough global memory.\n");
        fprintf(stderr, "Required: %.1f GB\n", (double)total/(1024.0*1024.0*1024.0));
        fprintf(stderr, "Available: %.1f GB\n", (double)global/(1024.0*1024.0*1024.0));
        exit(EXIT_FAILURE);
    }
}

void allocate_buffers(
    struct problem * problem, struct rankinfo * rankinfo,
    struct buffers * buffers)
{
    cl_int err;

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


void zero_buffer(struct context * context, cl_mem buffer, size_t offset, size_t size)
{
    cl_int err;
    err = clSetKernelArg(context->kernels.zero_buffer, 0, sizeof(cl_mem), &buffer);
    check_ocl(err, "Setting buffer zero kernel argument");
    err = clEnqueueNDRangeKernel(context->queue,
        context->kernels.zero_buffer,
        1, &offset, &size, NULL, 0, NULL, NULL);
    check_ocl(err, "Enqueueing buffer zero kernel");
}

void swap_angular_flux_buffers(struct buffers * buffers)
{
    for (int i = 0; i < 8; i++)
    {
        cl_mem tmp;
        tmp = buffers->angular_flux_in[i];
        buffers->angular_flux_in[i] = buffers->angular_flux_out[i];
        buffers->angular_flux_out[i] = tmp;
    }
}
