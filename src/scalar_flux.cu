
#include "scalar_flux.h"

void compute_scalar_flux(
    struct problem * problem,
    struct rankinfo * rankinfo,
    struct buffers * buffers,
    struct events * events
    )
{

    // get closest power of 2 to nang
    size_t power = 1 << (unsigned int)ceil(log2((double)problem->nang));

    dim3 blocks(problem->ng, ceil(rankinfo->nx*rankinfo->ny*rankinfo->nz/(float)BLOCK_SIZE_2D), 1);
    dim3 threads(power, BLOCK_SIZE_2D, 1);

    cudaEventRecord(events->scalar_flux_event_start);
    check_cuda("Recording scalar flux start event");

    reduce_flux<<< blocks, threads, sizeof(double)*power*BLOCK_SIZE_2D >>>(
        rankinfo->nx, rankinfo->ny, rankinfo->nz,
        problem->nang, problem->ng,
        buffers->angular_flux_in[0],
        buffers->angular_flux_in[1],
        buffers->angular_flux_in[2],
        buffers->angular_flux_in[3],
        buffers->angular_flux_in[4],
        buffers->angular_flux_in[5],
        buffers->angular_flux_in[6],
        buffers->angular_flux_in[7],
        buffers->angular_flux_out[0],
        buffers->angular_flux_out[1],
        buffers->angular_flux_out[2],
        buffers->angular_flux_out[3],
        buffers->angular_flux_out[4],
        buffers->angular_flux_out[5],
        buffers->angular_flux_out[6],
        buffers->angular_flux_out[7],
        buffers->velocity_delta, buffers->quad_weights,
        buffers->scalar_flux
    );
    check_cuda("Enqueueing scalar flux reduction kernel");

    cudaEventRecord(events->scalar_flux_event_stop);
    check_cuda("Recording scalar flux stop event");
}

void compute_scalar_flux_moments(
    struct problem * problem,
    struct rankinfo * rankinfo,
    struct buffers * buffers,
    struct events * events
    )
{

    // get closest power of 2 to nang
    size_t power = 1 << (unsigned int)ceil(log2((double)problem->nang));

    dim3 blocks(problem->ng, ceil(rankinfo->nx*rankinfo->ny*rankinfo->nz/(float)BLOCK_SIZE_2D), 1);
    dim3 threads(power, BLOCK_SIZE_2D, 1);

    cudaEventRecord(events->scalar_flux_moments_event_start);
    check_cuda("Recording scalar flux moments start event");

    reduce_flux_moments<<< blocks, threads, sizeof(double)*power*BLOCK_SIZE_2D >>>(
        rankinfo->nx, rankinfo->ny, rankinfo->nz,
        problem->nang, problem->ng, problem->cmom,
        buffers->angular_flux_in[0],
        buffers->angular_flux_in[1],
        buffers->angular_flux_in[2],
        buffers->angular_flux_in[3],
        buffers->angular_flux_in[4],
        buffers->angular_flux_in[5],
        buffers->angular_flux_in[6],
        buffers->angular_flux_in[7],
        buffers->angular_flux_out[0],
        buffers->angular_flux_out[1],
        buffers->angular_flux_out[2],
        buffers->angular_flux_out[3],
        buffers->angular_flux_out[4],
        buffers->angular_flux_out[5],
        buffers->angular_flux_out[6],
        buffers->angular_flux_out[7],
        buffers->velocity_delta, buffers->quad_weights,
        buffers->scat_coeff,
        buffers->scalar_flux_moments
    );
    check_cuda("Enqueueing scalar flux moments reduction kernel");

    cudaEventRecord(events->scalar_flux_moments_event_stop);
    check_cuda("Recording scalar flux moments stop event");
}


void copy_back_scalar_flux(
    struct problem *problem,
    struct rankinfo * rankinfo,
    struct buffers * buffers,
    double * scalar_flux
    )
{
    cudaMemcpy(scalar_flux, buffers->scalar_flux,
        sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz,
    cudaMemcpyDeviceToHost);
    check_cuda("Copying back scalar flux");
}
