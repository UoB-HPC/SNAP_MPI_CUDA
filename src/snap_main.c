
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <mpi.h>

#include "global.h"
#include "comms.h"
#include "input.h"
#include "problem.h"
#include "allocate.h"
#include "source.h"
#include "sweep.h"
#include "scalar_flux.h"
#include "convergence.h"
#include "population.h"
#include "profiler.h"

#include "ocl_global.h"
#include "ocl_buffers.h"

/** \mainpage
* SNAP-MPI is a cut down version of the SNAP mini-app which allows us to
* investigate MPI decomposition schemes with various node-level implementations.
* In particular, this code will allow:
* \li Flat MPI
* \li Hybrid MPI+OpenMP (For CPU and larger core counts)
* \li OpenCL
*
* The MPI scheme used is KBA, expending into hybrid-KBA.
*/

/** \brief Print out starting information */
void print_banner(void);

/** \brief Print out the input paramters */
void print_input(struct problem * problem);

/** \brief Print out OpenCL information */
void print_opencl_info(struct context * context);

/** \brief Print out the timing report */
void print_timing_report(struct timers * timers, struct problem * problem, unsigned int total_iterations);

#define MAX_INFO_STRING 256
#define STARS "********************************************************"

/** \brief Main function, contains iteration loops */
int main(int argc, char **argv)
{
    int mpi_err = MPI_Init(&argc, &argv);
    check_mpi(mpi_err, "MPI_Init");

    cl_int clerr;

    struct timers timers;
    timers.setup_time = wtime();

    int rank, size;
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    check_mpi(mpi_err, "Getting MPI rank");

    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    check_mpi(mpi_err, "Getting MPI size");

    struct problem problem;

    if (rank == 0)
    {
        print_banner();

        // Check for two files on CLI
        if (argc != 2)
        {
            fprintf(stderr, "Usage: ./snap snap.in\n");
            exit(-1);
        }
        read_input(argv[1], &problem);
        if ((problem.npex * problem.npey * problem.npez) != size)
        {
            fprintf(stderr, "Input error: wanted %d ranks but executing with %d\n", problem.npex*problem.npey*problem.npez, size);
            exit(-1);
        }
        check_decomposition(&problem);

    }

    // Set dx, dy, dz, dt values
    problem.dx = problem.lx / (double)problem.nx;
    problem.dy = problem.ly / (double)problem.ny;
    problem.dz = problem.lz / (double)problem.nz;
    problem.dt = problem.tf / (double)problem.nsteps;

    // Echo input file to screen
    if (rank == 0)
        print_input(&problem);

    // Broadcast the global variables
    broadcast_problem(&problem, rank);

    // Set up communication neighbours
    struct rankinfo rankinfo;
    setup_comms(&problem, &rankinfo);

    // Initlise the OpenCL
    struct context context;
    init_ocl(&context);
    if (rankinfo.rank == 0)
        print_opencl_info(&context);
    struct buffers buffers;
    check_device_memory_requirements(&problem, &rankinfo, &context);
    allocate_buffers(&problem, &rankinfo, &context, &buffers);

    // Allocate the problem arrays
    struct memory memory;
    allocate_memory(&problem, &rankinfo, &memory);


    // Set up problem
    init_quadrature_weights(&problem, &context, &buffers);
    calculate_cosine_coefficients(&problem, &context, &buffers, memory.mu, memory.eta, memory.xi);
    calculate_scattering_coefficients(&problem, &context, &buffers, memory.mu, memory.eta, memory.xi);
    init_material_data(&problem, &context, &buffers, memory.mat_cross_section);
    init_fixed_source(&problem, &rankinfo, &context, &buffers);
    init_scattering_matrix(&problem, &context, &buffers, memory.mat_cross_section);
    init_velocities(&problem, &context, &buffers);

    struct plane* planes;
    unsigned int num_planes;
    init_planes(&planes, &num_planes, &rankinfo);
    copy_planes(planes, num_planes, &context, &buffers);

    // Zero out the angular flux buffers
    for (int oct = 0; oct < 8; oct++)
    {
        zero_buffer(&context, buffers.angular_flux_in[oct], 0, problem.nang*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
        zero_buffer(&context, buffers.angular_flux_out[oct], 0, problem.nang*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
    }

    // Zero out the outer source, because later moments are +=
    zero_buffer(&context, buffers.outer_source, 0, problem.cmom*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);

    clerr = clFinish(context.queue);
    check_ocl(clerr, "Finish queue at end of setup");

    if (rankinfo.rank == 0)
        timers.setup_time = wtime() - timers.setup_time;

    bool innerdone, outerdone;

    // Timers
    if (rankinfo.rank == 0)
        timers.simulation_time = wtime();

    if (rankinfo.rank == 0)
    {
        printf("%s\n", STARS);
        printf("  Iteration Monitor\n");
        printf("%s\n", STARS);
    }

    unsigned int total_iterations = 0;

    //----------------------------------------------
    // Timestep loop
    //----------------------------------------------
    for (unsigned int t = 0; t < problem.nsteps; t++)
    {
        if (rankinfo.rank == 0)
        {
            printf(" Timestep %d\n", t);
            printf("   %-10s %-15s %-10s\n", "Outer", "Difference", "Inners");
        }

        // Zero out the scalar flux and flux moments
        zero_buffer(&context, buffers.scalar_flux, 0, problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
        if (problem.cmom-1 > 0)
            zero_buffer(&context, buffers.scalar_flux_moments, 0, (problem.cmom-1)*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);

        // Swap angluar flux pointers (not for the first timestep)
        if (t > 0)
            swap_angular_flux_buffers(&buffers);

        //----------------------------------------------
        // Outers
        //----------------------------------------------
        for (unsigned int o = 0; o < problem.oitm; o++)
        {
            init_velocity_delta(&problem, &context, &buffers);
            calculate_dd_coefficients(&problem, &context, &buffers);
            calculate_denominator(&problem, &rankinfo, &context, &buffers);

            compute_outer_source(&problem, &rankinfo, &context, &buffers);

            // Get the scalar flux back
            copy_back_scalar_flux(&problem, &rankinfo, &context, &buffers, memory.old_outer_scalar_flux, CL_FALSE);

            //----------------------------------------------
            // Inners
            //----------------------------------------------
            unsigned int i;
            for (i = 0; i < problem.iitm; i++)
            {
                compute_inner_source(&problem, &rankinfo, &context, &buffers);

                // Get the scalar flux back
                copy_back_scalar_flux(&problem, &rankinfo, &context, &buffers, memory.old_inner_scalar_flux, CL_FALSE);


                double sweep_tick;
                if (profiling && rankinfo.rank == 0)
                {
                    // We must wait for the transfer to finish before we enqueue the next transfer,
                    // or MPI_Recv to get accurate timings
                    clerr = clFinish(context.queue);
                    check_ocl(clerr, "Finish queue just before sweep for profiling");
                    sweep_tick = wtime();
                }

                // Sweep each octant in turn
                int octant = 0;
                for (int istep = -1; istep < 2; istep += 2)
                    for (int jstep = -1; jstep < 2; jstep += 2)
                        for (int kstep = -1; kstep < 2; kstep += 2)
                        {
                            // Zero the z buffer every octant - we just do KBA
                            zero_buffer(&context, buffers.flux_k, 0, problem.nang*problem.ng*rankinfo.nx*rankinfo.ny);

                            // Receive your first XY plane and/or boundary values
                            recv_boundaries(0, octant, istep, jstep, kstep, &problem, &rankinfo, &memory, &context, &buffers);

                            // Complete the first XY-plane apart from the final corner cell
                            unsigned int p;
                            for (p = 0; p < rankinfo.nx+rankinfo.ny - 2; p++)
                            {
                                sweep_plane(octant, istep, jstep, kstep, p, planes, &problem, &rankinfo, &context, &buffers);
                            }

                            for (unsigned int z_pos = 0; z_pos < rankinfo.nz; z_pos += problem.chunk)
                            {
                                if (z_pos != 0)
                                {
                                    // Receive the next plane, but we already have the first one from above
                                    recv_boundaries(z_pos, octant, istep, jstep, kstep, &problem, &rankinfo, &memory, &context, &buffers);
                                }

                                // Sweep a chunk == chunk more planes
                                for (unsigned int c = 0; c <  problem.chunk; c++)
                                {
                                    sweep_plane(octant, istep, jstep, kstep, p, planes, &problem, &rankinfo, &context, &buffers);
                                    p++;
                                }

                                send_boundaries(z_pos, octant, istep, jstep, kstep, &problem, &rankinfo, &memory, &context, &buffers);
                            }

                            octant += 1;
                        }

                if (profiling && rankinfo.rank == 0)
                {
                    // The last send boundaries is either a blocking read of blocking MPI_Send,
                    // so we know everything in the queue is done
                    timers.sweep_time += wtime() - sweep_tick;
                }

                // Compute the Scalar Flux
                compute_scalar_flux(&problem, &rankinfo, &context, &buffers);
                if (problem.cmom-1 > 0)
                    compute_scalar_flux_moments(&problem, &rankinfo, &context, &buffers);

                // Get the new scalar flux back and check inner convergence
                copy_back_scalar_flux(&problem, &rankinfo, &context, &buffers, memory.scalar_flux, CL_TRUE);

                double conv_tick = wtime();

                innerdone = inner_convergence(&problem, &rankinfo, &memory);

                if (profiling && rankinfo.rank == 0)
                    timers.convergence_time += wtime() - conv_tick;

                // Do any profiler updates for timings
                if (rankinfo.rank == 0)
                    inner_profiler(&timers, &problem);

                if (innerdone)
                    break;

            }
            //----------------------------------------------
            // End of Inners
            //----------------------------------------------

            // Check outer convergence
            // We don't need to copy back the new scalar flux again as it won't have changed from the last inner
            double max_outer_diff;
            double conv_tick = wtime();
            outerdone = outer_convergence(&problem, &rankinfo, &memory, &max_outer_diff) && innerdone;

            if (profiling && rankinfo.rank == 0)
                timers.convergence_time += wtime() - conv_tick;

            total_iterations += i;

            if (rankinfo.rank == 0)
                printf("     %-9u %-15lf %-10u\n", o, max_outer_diff, i);

            // Do any profiler updates for timings
            if (rankinfo.rank == 0)
                outer_profiler(&timers);

            if (outerdone)
                break;

        }
        //----------------------------------------------
        // End of Outers
        //----------------------------------------------

        // Exit the time loop early if outer not converged
        if (!outerdone)
        {
            if (rankinfo.rank == 0)
                printf(" * Stopping because not converged *\n");
            break;
        }

        // Calculate particle population and print out the value
        double population;
        calculate_population(&problem, &rankinfo, &memory, &population);
        if (rankinfo.rank == 0)
        {
            // Get exponent of outer convergence criteria
            int places;
            frexp(100.0 * problem.epsi, &places);
            places = ceil(fabs(places / log2(10)));
            char format[100];
            sprintf(format, "   Population: %%.%dlf\n", places);
            printf("\n");
            printf(format, population);
            printf("\n");
        }

    }
    //----------------------------------------------
    // End of Timestep
    //----------------------------------------------

    clerr = clFinish(context.queue);
    check_ocl(clerr, "Finishing queue before simulation end");

    if (rankinfo.rank == 0)
    {
        timers.simulation_time = wtime() - timers.simulation_time;

        print_timing_report(&timers, &problem, total_iterations);
    }

    free_memory(&memory);

    release_context(&context);
    finish_comms();
}

void print_banner(void)
{
    printf("\n");
    printf(" SNAP: SN (Discrete Ordinates) Application Proxy\n");
    printf(" MPI+OpenCL port\n");
    time_t rawtime;
    struct tm * timeinfo;
    char timestring[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(timestring, 80, "%c", timeinfo);
    printf(" Run on %s\n", timestring);
    printf("\n");
}

void print_input(struct problem * problem)
{
    printf("\n%s\n", STARS);
    printf(  "  Input Parameters\n");
    printf(  "%s\n", STARS);

    printf(" Geometry\n");
    printf("   %-30s %.3lf x %.3lf x %.3lf\n", "Problem size:", problem->lx, problem->ly, problem->lz);
    printf("   %-30s %5u x %5u x %5u\n", "Cells:", problem->nx, problem->ny, problem->nz);
    printf("   %-30s %.3lf x %.3lf x %.3lf\n", "Cell size:", problem->dx, problem->dy, problem->dz);
    printf("\n");

    printf(" Discrete Ordinates\n");
    printf("   %-30s %u\n", "Angles per octant:", problem->nang);
    printf("   %-30s %u\n", "Moments:", problem->nmom);
    printf("   %-30s %u\n", "\"Computational\" moments:", problem->cmom);
    printf("\n");

    printf(" Energy groups\n");
    printf("   %-30s %u\n", "Number of groups:", problem->ng);
    printf("\n");

    printf(" Timesteps\n");
    printf("   %-30s %u\n", "Timesteps:", problem->nsteps);
    printf("   %-30s %.3lf\n", "Simulation time:", problem->tf);
    printf("   %-30s %.3lf\n", "Time delta:", problem->dt);
    printf("\n");

    printf(" Iterations\n");
    printf("   %-30s %u\n", "Max outers per timestep:", problem->oitm);
    printf("   %-30s %u\n", "Max inners per outer:", problem->iitm);

    printf("   Stopping criteria\n");
    printf("     %-28s %.2E\n", "Inner convergence:", problem->epsi);
    printf("     %-28s %.2E\n", "Outer convergence:", 100.0*problem->epsi);
    printf("\n");

    printf(" MPI decomposition\n");
    printf("   %-30s %u x %u x %u\n", "Rank layout:", problem->npex, problem->npey, problem->npez);
    printf("   %-30s %u\n", "Chunk size:", problem->chunk);
    printf("\n");

}

void print_opencl_info(struct context * context)
{
    cl_int err;
    char info_string[MAX_INFO_STRING];

    printf("\n%s\n", STARS);
    printf("  OpenCL Information\n");
    printf("%s\n", STARS);

    // Print out device name
    err = clGetDeviceInfo(context->device, CL_DEVICE_NAME, sizeof(info_string), info_string, NULL);
    check_ocl(err, "Getting device name");
    printf(" Device\n");
    printf("   %s\n", info_string);
    printf("\n");

    // Driver version
    err = clGetDeviceInfo(context->device, CL_DRIVER_VERSION, sizeof(info_string), info_string, NULL);
    check_ocl(err, "Getting driver version");
    printf(" Driver\n");
    printf("   %s\n", info_string);
    printf("\n");
}

void print_timing_report(struct timers * timers, struct problem * problem, unsigned int total_iterations)
{
    printf("\n%s\n", STARS);
    printf(  "  Timing Report\n");
    printf(  "%s\n", STARS);

    printf(" %-30s %6.3lfs\n", "Setup", timers->setup_time);
    if (profiling)
    {
        printf(" %-30s %6.3lfs\n", "Outer source", timers->outer_source_time);
        printf(" %-30s %6.3lfs\n", "Outer parameters", timers->outer_params_time);
        printf(" %-30s %6.3lfs\n", "Inner source", timers->inner_source_time);
        printf(" %-30s %6.3lfs\n", "Sweeps", timers->sweep_time);
        printf(" %-30s %6.3lfs\n", "Scalar flux reductions", timers->reduction_time);
        printf(" %-30s %6.3lfs\n", "Convergence checking", timers->convergence_time);
        printf(" %-30s %6.3lfs\n", "Other", timers->simulation_time - timers->outer_source_time - timers->outer_params_time - timers->inner_source_time - timers->sweep_time - timers->reduction_time - timers->convergence_time);
        }
        printf(" %-30s %6.3lfs\n", "Total simulation", timers->simulation_time);

        printf("\n");
        printf(" %-30s %6.3lfns\n", "Grind time",
            1.0E9 * timers->simulation_time /
            (double)(problem->nx*problem->ny*problem->nz*problem->nang*8*problem->ng*total_iterations)
            );

        printf( "%s\n", STARS);

}

