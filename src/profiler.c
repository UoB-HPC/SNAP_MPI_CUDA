
#include "profiler.h"

double wtime(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1.0E-6;
}


void outer_profiler(struct timers * timers)
{
    if (!profiling)
        return;


    // Times are in milliseconds
    float time;

    // Get outer souce update times
    cudaEventElapsedTime(&time, outer_source_event_start, outer_source_event_stop);
    check_cuda("Getting outer source time");
    timers->outer_source_time += (double)(time) * 1.0E-3;

/*
    // Get outer parameter times
    // Start is velocity delta start, end is denominator end
    err = clGetEventProfilingInfo(velocity_delta_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
    check_ocl(err, "Getting velocity delta start time");
    err = clGetEventProfilingInfo(denominator_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
    check_ocl(err, "Getting denominator end time");
    timers->outer_params_time += (double)(tock - tick) * 1.0E-9;
*/
}



void inner_profiler(struct timers * timers, struct problem * problem)
{
    if (!profiling)
        return;

    // Times are in milliseconds
    float time;

    // Get inner source update times
    cudaEventElapsedTime(&time, inner_source_event_start, inner_source_event_stop);
    check_cuda("Cetting inner source time");
    timers->inner_source_time += (double)(time) * 1.0E-3;

/*
    // Get scalar flux reduction times
    err = clGetEventProfilingInfo(scalar_flux_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
    check_ocl(err, "Getting scalar flux start time");
    err = clGetEventProfilingInfo(scalar_flux_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
    check_ocl(err, "Getting scalar flux end time");
    timers->reduction_time += (double)(tock - tick) * 1.0E-9;
    if (problem->cmom-1 > 0)
    {
        err = clGetEventProfilingInfo(scalar_flux_moments_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
        check_ocl(err, "Getting scalar flux moments start time");
        err = clGetEventProfilingInfo(scalar_flux_moments_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
        check_ocl(err, "Getting scalar flux moments end time");
        timers->reduction_time += (double)(tock - tick) * 1.0E-9;
    }
*/
}

/*

void chunk_profiler(struct timers * timers)
{
    if (!profiling)
        return;

    cl_int err;

    // Times are in nanoseconds
    cl_ulong tick, tock;

    // Get recv writes
    if (flux_i_write_event)
    {
        err = clGetEventProfilingInfo(flux_i_write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
        check_ocl(err, "Getting flux i write start time");
        err = clGetEventProfilingInfo(flux_i_write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
        check_ocl(err, "Getting flux i write stop time");
        timers->sweep_transfer_time += (double)(tock - tick) * 1.0E-9;
    }

    if (flux_j_write_event)
    {
        err = clGetEventProfilingInfo(flux_j_write_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
        check_ocl(err, "Getting flux j write start time");
        err = clGetEventProfilingInfo(flux_j_write_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
        check_ocl(err, "Getting flux j write stop time");
        timers->sweep_transfer_time += (double)(tock - tick) * 1.0E-9;
    }

    // Get send reads
    if (flux_i_read_event)
    {
        err = clGetEventProfilingInfo(flux_i_read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
        check_ocl(err, "Getting flux i read start time");
        err = clGetEventProfilingInfo(flux_i_read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
        check_ocl(err, "Getting flux i read stop time");
        timers->sweep_transfer_time += (double)(tock - tick) * 1.0E-9;
    }

    if (flux_j_read_event)
    {
        err = clGetEventProfilingInfo(flux_j_read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &tick, NULL);
        check_ocl(err, "Getting flux j read start time");
        err = clGetEventProfilingInfo(flux_j_read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &tock, NULL);
        check_ocl(err, "Getting flux j read stop time");
        timers->sweep_transfer_time += (double)(tock - tick) * 1.0E-9;
    }
}

*/

void create_events(void)
{
    cudaEventCreate(&outer_source_event_start);
    check_cuda("Creating outer source event start");
    cudaEventCreate(&outer_source_event_stop);
    check_cuda("Creating outer source event stop");
    cudaEventCreate(&inner_source_event_start);
    check_cuda("Creating inner source event start");
    cudaEventCreate(&inner_source_event_stop);
    check_cuda("Creating inner source event stop");

    cudaEventCreate(&scalar_flux_event_start);
    check_cuda("Creating scalar flux event start");
    cudaEventCreate(&scalar_flux_event_stop);
    check_cuda("Creating scalar flux event stop");
    cudaEventCreate(&scalar_flux_moments_event_start);
    check_cuda("Creating scalar flux moments event start");
    cudaEventCreate(&scalar_flux_moments_event_stop);
    check_cuda("Creating scalar flux moments event stop");

    cudaEventCreate(&velocity_delta_event_start);
    check_cuda("Creating velocity delta event start");
    cudaEventCreate(&velocity_delta_event_stop);
    check_cuda("Creating velocity delta event stop");
    cudaEventCreate(&denominator_event_start);
    check_cuda("Creating denoninator event start");
    cudaEventCreate(&denominator_event_stop);
    check_cuda("Creating denoninator event stop");

    cudaEventCreate(&flux_i_read_event_start);
    check_cuda("Creating flux i read event start");
    cudaEventCreate(&flux_i_read_event_stop);
    check_cuda("Creating flux i read event stop");
    cudaEventCreate(&flux_j_read_event_start);
    check_cuda("Creating flux j read event start");
    cudaEventCreate(&flux_j_read_event_stop);
    check_cuda("Creating flux j read event stop");
    cudaEventCreate(&flux_i_write_event_start);
    check_cuda("Creating flux i write event start");
    cudaEventCreate(&flux_i_write_event_stop);
    check_cuda("Creating flux i write event stop");
    cudaEventCreate(&flux_j_write_event_start);
    check_cuda("Creating flux j write event start");
    cudaEventCreate(&flux_j_write_event_stop);
    check_cuda("Creating flux j write event stop");
}


void destroy_events(void)
{
    cudaEventDestroy(outer_source_event_start);
    check_cuda("Creating outer source event start");
    cudaEventDestroy(outer_source_event_stop);
    check_cuda("Creating outer source event stop");
    cudaEventDestroy(inner_source_event_start);
    check_cuda("Creating inner source event start");
    cudaEventDestroy(inner_source_event_stop);
    check_cuda("Creating inner source event stop");

    cudaEventDestroy(scalar_flux_event_start);
    check_cuda("Creating scalar flux event start");
    cudaEventDestroy(scalar_flux_event_stop);
    check_cuda("Creating scalar flux event stop");
    cudaEventDestroy(scalar_flux_moments_event_start);
    check_cuda("Creating scalar flux moments event start");
    cudaEventDestroy(scalar_flux_moments_event_stop);
    check_cuda("Creating scalar flux moments event stop");

    cudaEventDestroy(velocity_delta_event_start);
    check_cuda("Creating velocity delta event start");
    cudaEventDestroy(velocity_delta_event_stop);
    check_cuda("Creating velocity delta event stop");
    cudaEventDestroy(denominator_event_start);
    check_cuda("Creating denoninator event start");
    cudaEventDestroy(denominator_event_stop);
    check_cuda("Creating denoninator event stop");

    cudaEventDestroy(flux_i_read_event_start);
    check_cuda("Creating flux i read event start");
    cudaEventDestroy(flux_i_read_event_stop);
    check_cuda("Creating flux i read event stop");
    cudaEventDestroy(flux_j_read_event_start);
    check_cuda("Creating flux j read event start");
    cudaEventDestroy(flux_j_read_event_stop);
    check_cuda("Creating flux j read event stop");
    cudaEventDestroy(flux_i_write_event_start);
    check_cuda("Creating flux i write event start");
    cudaEventDestroy(flux_i_write_event_stop);
    check_cuda("Creating flux i write event stop");
    cudaEventDestroy(flux_j_write_event_start);
    check_cuda("Creating flux j write event start");
    cudaEventDestroy(flux_j_write_event_stop);
    check_cuda("Creating flux j write event stop");
}

