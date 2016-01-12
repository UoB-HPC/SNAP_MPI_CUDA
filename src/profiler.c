
#include "profiler.h"

double wtime(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1.0E-6;
}


void outer_profiler(struct timers * timers, struct events * events)
{
    if (!profiling)
        return;


    // Times are in milliseconds
    float time;

    // Get outer souce update times
    cudaEventElapsedTime(&time, events->outer_source_event_start, events->outer_source_event_stop);
    check_cuda("Getting outer source time");
    timers->outer_source_time += (double)(time) * 1.0E-3;

    // Get outer parameter times
    // Start is velocity delta start, end is denominator end
    cudaEventElapsedTime(&time, events->velocity_delta_event_start, events->denominator_event_stop);
    check_cuda("Getting outer parameters time");
    timers->outer_params_time += (double)(time) * 1.0E-3;
}



void inner_profiler(struct timers * timers, struct problem * problem, struct events * events)
{
    if (!profiling)
        return;

    // Times are in milliseconds
    float time;

    // Get inner source update times
    cudaEventElapsedTime(&time, events->inner_source_event_start, events->inner_source_event_stop);
    check_cuda("Getting inner source time");
    timers->inner_source_time += (double)(time) * 1.0E-3;

    // Get scalar flux reduction times
    cudaEventElapsedTime(&time, events->scalar_flux_event_start, events->scalar_flux_event_stop);
    check_cuda("Getting scalar flux time");
    timers->reduction_time += (double)(time) * 1.0E-3;
    if (problem->cmom-1 > 0)
    {
        cudaEventElapsedTime(&time, events->scalar_flux_moments_event_start, events->scalar_flux_moments_event_stop);
        check_cuda("Getting scalar flux moments time");
        timers->reduction_time += (double)(time) * 1.0E-3;
    }
}


void chunk_profiler(struct timers * timers, struct events * events)
{
    if (!profiling)
        return;

    cudaError_t err;
    // Times are in nanoseconds
    float time;

    // Get recv writes
    cudaEventElapsedTime(&time, events->flux_i_write_event_start, events->flux_i_write_event_stop);
    err = cudaGetLastError();
    if (err != cudaErrorInvalidResourceHandle)
    {
        check_cuda("Getting flux i write time");
        timers->sweep_transfer_time += (double)(time) * 1.0E-3;
    }

    cudaEventElapsedTime(&time, events->flux_j_write_event_start, events->flux_j_write_event_stop);
    err = cudaGetLastError();
    if (err != cudaErrorInvalidResourceHandle)
    {
        check_cuda("Getting flux j write time");
        timers->sweep_transfer_time += (double)(time) * 1.0E-3;
    }

    // Get send reads
    cudaEventElapsedTime(&time, events->flux_i_read_event_start, events->flux_i_read_event_stop);
    err = cudaGetLastError();
    if (err != cudaErrorInvalidResourceHandle)
    {
        check_cuda("Getting flux i read time");
        timers->sweep_transfer_time += (double)(time) * 1.0E-3;
    }

    cudaEventElapsedTime(&time, events->flux_j_read_event_start, events->flux_j_read_event_stop);
    err = cudaGetLastError();
    if (err != cudaErrorInvalidResourceHandle)
    {
        check_cuda("Getting flux j read time");
        timers->sweep_transfer_time += (double)(time) * 1.0E-3;
    }

}


void create_events(struct events * events)
{
    cudaEventCreate(&events->outer_source_event_start);
    check_cuda("Creating outer source event start");
    cudaEventCreate(&events->outer_source_event_stop);
    check_cuda("Creating outer source event stop");
    cudaEventCreate(&events->inner_source_event_start);
    check_cuda("Creating inner source event start");
    cudaEventCreate(&events->inner_source_event_stop);
    check_cuda("Creating inner source event stop");

    cudaEventCreate(&events->scalar_flux_event_start);
    check_cuda("Creating scalar flux event start");
    cudaEventCreate(&events->scalar_flux_event_stop);
    check_cuda("Creating scalar flux event stop");
    cudaEventCreate(&events->scalar_flux_moments_event_start);
    check_cuda("Creating scalar flux moments event start");
    cudaEventCreate(&events->scalar_flux_moments_event_stop);
    check_cuda("Creating scalar flux moments event stop");

    cudaEventCreate(&events->velocity_delta_event_start);
    check_cuda("Creating velocity delta event start");
    cudaEventCreate(&events->denominator_event_stop);
    check_cuda("Creating denoninator event stop");

    cudaEventCreate(&events->flux_i_read_event_start);
    check_cuda("Creating flux i read event start");
    cudaEventCreate(&events->flux_i_read_event_stop);
    check_cuda("Creating flux i read event stop");
    cudaEventCreate(&events->flux_j_read_event_start);
    check_cuda("Creating flux j read event start");
    cudaEventCreate(&events->flux_j_read_event_stop);
    check_cuda("Creating flux j read event stop");
    cudaEventCreate(&events->flux_i_write_event_start);
    check_cuda("Creating flux i write event start");
    cudaEventCreate(&events->flux_i_write_event_stop);
    check_cuda("Creating flux i write event stop");
    cudaEventCreate(&events->flux_j_write_event_start);
    check_cuda("Creating flux j write event start");
    cudaEventCreate(&events->flux_j_write_event_stop);
    check_cuda("Creating flux j write event stop");
}


void destroy_events(struct events * events)
{
    cudaEventDestroy(events->outer_source_event_start);
    check_cuda("Creating outer source event start");
    cudaEventDestroy(events->outer_source_event_stop);
    check_cuda("Creating outer source event stop");
    cudaEventDestroy(events->inner_source_event_start);
    check_cuda("Creating inner source event start");
    cudaEventDestroy(events->inner_source_event_stop);
    check_cuda("Creating inner source event stop");

    cudaEventDestroy(events->scalar_flux_event_start);
    check_cuda("Creating scalar flux event start");
    cudaEventDestroy(events->scalar_flux_event_stop);
    check_cuda("Creating scalar flux event stop");
    cudaEventDestroy(events->scalar_flux_moments_event_start);
    check_cuda("Creating scalar flux moments event start");
    cudaEventDestroy(events->scalar_flux_moments_event_stop);
    check_cuda("Creating scalar flux moments event stop");

    cudaEventDestroy(events->velocity_delta_event_start);
    check_cuda("Creating velocity delta event start");
    check_cuda("Creating denoninator event start");
    cudaEventDestroy(events->denominator_event_stop);
    check_cuda("Creating denoninator event stop");

    cudaEventDestroy(events->flux_i_read_event_start);
    check_cuda("Creating flux i read event start");
    cudaEventDestroy(events->flux_i_read_event_stop);
    check_cuda("Creating flux i read event stop");
    cudaEventDestroy(events->flux_j_read_event_start);
    check_cuda("Creating flux j read event start");
    cudaEventDestroy(events->flux_j_read_event_stop);
    check_cuda("Creating flux j read event stop");
    cudaEventDestroy(events->flux_i_write_event_start);
    check_cuda("Creating flux i write event start");
    cudaEventDestroy(events->flux_i_write_event_stop);
    check_cuda("Creating flux i write event stop");
    cudaEventDestroy(events->flux_j_write_event_start);
    check_cuda("Creating flux j write event start");
    cudaEventDestroy(events->flux_j_write_event_stop);
    check_cuda("Creating flux j write event stop");
}

