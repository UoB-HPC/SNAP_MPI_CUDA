
#include "ocl_buffers.h"


void check_device_memory_requirements(
    struct problem * problem, struct rankinfo * rankinfo,
    struct context * context)
{
    cl_int err;
    cl_ulong global;
    err = clGetDeviceInfo(context->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global, NULL);

    cl_ulong total = 0;
    // Add up the memory requirements, in bytes.
    total += problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    total += problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    total += problem->nang*problem->ng*rankinfo->ny*rankinfo->nz;
    total += problem->nang*problem->ng*rankinfo->nx*rankinfo->nz;
    total += problem->nang*problem->ng*rankinfo->nx*rankinfo->ny;
    total += problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    if (problem->cmom-1 == 0)
        total += problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    else
        total += (problem->cmom-1)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    total += problem->ng;
    total += problem->ng;
    total += problem->ng;
    total += problem->ng;
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
    struct context * context, struct buffers * buffers)
{
    cl_int err;

    // Angular flux
    for (int i = 0; i < 8; i++)
    {
        buffers->angular_flux_in[i] = clCreateBuffer(
            context->context,
            CL_MEM_READ_WRITE,
            sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz,
            NULL, &err);
        check_ocl(err, "Creating an angular flux in buffer");

        buffers->angular_flux_out[i] = clCreateBuffer(
            context->context,
            CL_MEM_READ_WRITE,
            sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz,
            NULL, &err);
        check_ocl(err, "Creating an angular flux out buffer");
    }

    // Edge fluxes
    buffers->flux_i = clCreateBuffer(context->context, CL_MEM_READ_WRITE,
        sizeof(double)*problem->nang*problem->ng*rankinfo->ny*rankinfo->nz,
        NULL, &err);
    check_ocl(err, "Creating flux_i buffer");

    buffers->flux_j = clCreateBuffer(context->context, CL_MEM_READ_WRITE,
        sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->nz,
        NULL, &err);
    check_ocl(err, "Creating flux_j buffer");

    buffers->flux_k = clCreateBuffer(context->context, CL_MEM_READ_WRITE,
        sizeof(double)*problem->nang*problem->ng*rankinfo->nx*rankinfo->ny,
        NULL, &err);
    check_ocl(err, "Creating flux_k buffer");

    // Scalar flux
    buffers->scalar_flux = clCreateBuffer(context->context, CL_MEM_READ_WRITE,
        sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz,
        NULL, &err);
    check_ocl(err, "Creating scalar flux buffer");

    size_t scalar_moments_buffer_size;
    if (problem->cmom-1 == 0)
        scalar_moments_buffer_size = sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    else
        scalar_moments_buffer_size = sizeof(double)*(problem->cmom-1)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz;
    buffers->scalar_flux_moments = clCreateBuffer(context->context, CL_MEM_READ_WRITE,
        scalar_moments_buffer_size, NULL, &err);
    check_ocl(err, "Creating scalar flux moments buffer");

    buffers->quad_weights = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(double)*problem->ng, NULL, &err);
    check_ocl(err, "Crearing quadrature weights buffer");
    buffers->mu = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(double)*problem->ng, NULL, &err);
    check_ocl(err, "Crearing mu cosine buffer");
    buffers->eta = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(double)*problem->ng, NULL, &err);
    check_ocl(err, "Crearing eta cosine buffer");
    buffers->xi = clCreateBuffer(context->context, CL_MEM_READ_ONLY, sizeof(double)*problem->ng, NULL, &err);
    check_ocl(err, "Crearing xi cosine buffer");

}
