
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

#include "cuda_global.h"
#include "cuda_buffers.h"

double sweep_mpi_time = 0.0;
double sweep_mpi_recv_time = 0.0;

/** \mainpage
* SNAP-MPI is a cut down version of the SNAP mini-app which allows us to
* investigate MPI decomposition schemes with CUDA for node-level computation.
*
* The MPI scheme used is KBA, expanding into hybrid-KBA.
*/

/** \brief Print out starting information */
void print_banner(void);

/** \brief Print out the input paramters */
void print_input(struct problem * problem);

/** \brief Print out CUDA information */
void print_cuda_info(void);

/** \brief Print out the timing report */
void print_timing_report(struct timers * timers, struct problem * problem, unsigned int total_iterations);

#define MAX_INFO_STRING 256
#define STARS "********************************************************"

/** \brief Main function, contains iteration loops */
int main(int argc, char **argv)
{
    int mpi_err = MPI_Init(&argc, &argv);
    check_mpi(mpi_err, "MPI_Init");

    struct timers timers;
    zero_timers(&timers);
    timers.setup_time = wtime();

    struct events events;
    create_events(&events);

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
            exit(EXIT_FAILURE);
        }
        read_input(argv[1], &problem);
        if ((problem.npex * problem.npey * problem.npez) != size)
        {
            fprintf(stderr, "Input error: wanted %d ranks but executing with %d\n", problem.npex*problem.npey*problem.npez, size);
            exit(EXIT_FAILURE);
        }
        check_decomposition(&problem);

    }

    // Set dx, dy, dz, dt values
    problem.dx = problem.lx / (double)problem.nx;
    problem.dy = problem.ly / (double)problem.ny;
    problem.dz = problem.lz / (double)problem.nz;
    problem.dt = problem.tf / (double)problem.nsteps;

    // Broadcast the global variables
    broadcast_problem(&problem, rank);

    // Echo input file to screen
    if (rank == 0)
        print_input(&problem);

    // Set up communication neighbours
    struct rankinfo rankinfo;
    setup_comms(&problem, &rankinfo);

    // Initlise the CUDA
    if (rankinfo.rank == 0)
        print_cuda_info();
    struct buffers buffers;
    allocate_buffers(&problem, &rankinfo, &buffers);

    // Allocate the problem arrays
    struct memory memory;
    allocate_memory(&problem, &rankinfo, &memory);


    // Set up problem
    init_quadrature_weights(&problem, &buffers);
    calculate_cosine_coefficients(&problem, &buffers, memory.mu, memory.eta, memory.xi);
    calculate_scattering_coefficients(&problem, &buffers, memory.mu, memory.eta, memory.xi);
    init_material_data(&problem, &buffers, memory.mat_cross_section);
    init_fixed_source(&problem, &rankinfo, &buffers);
    init_scattering_matrix(&problem, &buffers, memory.mat_cross_section);
    init_velocities(&problem, memory.velocities, &buffers);

    struct plane* planes;
    unsigned int num_planes;
    init_planes(&planes, &num_planes, &problem, &rankinfo);
    copy_planes(planes, num_planes, &buffers);

    // Zero out the angular flux buffers
    for (int oct = 0; oct < 8; oct++)
    {
        cudaMemset(buffers.angular_flux_in[oct], (int)0.0,
            sizeof(double)*problem.nang*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
        check_cuda("Zeroing angular flux in");
        cudaMemset(buffers.angular_flux_out[oct], (int)0.0,
            sizeof(double)*problem.nang*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
        check_cuda("Zeroing angular flux out");
    }

    // Zero out the outer source, because later moments are +=
    cudaMemset(buffers.outer_source, (int)0.0, sizeof(double)*problem.cmom*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
    check_cuda("Zeroing outer source");


    cudaDeviceSynchronize();
    check_cuda("Finish queue at end of setup");

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
        cudaMemset(buffers.scalar_flux, (int)0.0, sizeof(double)*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
        check_cuda("Zero scalar flux");
        if (problem.cmom-1 > 0)
        {
            cudaMemset(buffers.scalar_flux_moments, (int)0.0, sizeof(double)*(problem.cmom-1)*problem.ng*rankinfo.nx*rankinfo.ny*rankinfo.nz);
            check_cuda("Zero scalar flux moments");
        }


        // Swap angluar flux pointers (not for the first timestep)
        if (t > 0)
            swap_angular_flux_buffers(&buffers);

        //----------------------------------------------
        // Outers
        //----------------------------------------------
        for (unsigned int o = 0; o < problem.oitm; o++)
        {
            init_velocity_delta(&problem, &buffers, &events);
            calculate_dd_coefficients(&problem, &buffers, &events);

            compute_outer_source(&problem, &rankinfo, &buffers, &events);

            // Get the scalar flux back
            copy_back_scalar_flux(&problem, &rankinfo, &buffers, memory.old_outer_scalar_flux);

            //----------------------------------------------
            // Inners
            //----------------------------------------------
            unsigned int i;
            for (i = 0; i < problem.iitm; i++)
            {
                compute_inner_source(&problem, &rankinfo, &buffers, &events);

                // Get the scalar flux back
                copy_back_scalar_flux(&problem, &rankinfo, &buffers, memory.old_inner_scalar_flux);

                double sweep_tick;
                if (profiling && rankinfo.rank == 0)
                {
                    // We must wait for the transfer to finish before we enqueue the next transfer,
                    // or MPI_Recv to get accurate timings
                    cudaDeviceSynchronize();
                    check_cuda("Finish queue just before sweep for profiling");
                    sweep_tick = wtime();
                }

                // Sweep each octant in turn
                int octant = 0;
                for (int istep = -1; istep < 2; istep += 2)
                    for (int jstep = -1; jstep < 2; jstep += 2)
                        for (int kstep = -1; kstep < 2; kstep += 2)
                        {
                            // Zero the z buffer every octant - we just do KBA
                            cudaMemset(buffers.flux_k, (int)0.0, sizeof(double)*problem.nang*problem.ng*rankinfo.nx*rankinfo.ny);
                            check_cuda("Setting flux k to zero");

                            for (unsigned int z_pos = 0; z_pos < rankinfo.nz; z_pos += problem.chunk)
                            {
                                double tick = wtime();
                                recv_boundaries(z_pos, octant, istep, jstep, kstep, &problem, &rankinfo, &memory, &buffers, &events);
                                sweep_mpi_recv_time += wtime() - tick;
                                for (unsigned int p = 0; p < num_planes; p++)
                                {
                                    sweep_plane(z_pos, octant, istep, jstep, kstep, p, planes, &problem, &rankinfo, &buffers);
                                }
                                send_boundaries(z_pos, octant, istep, jstep, kstep, &problem, &rankinfo, &memory, &buffers, &events);
                            }

                            if (profiling && rankinfo.rank == 0)
                                chunk_profiler(&timers, &events);

                            octant += 1;
                        }

                if (profiling && rankinfo.rank == 0)
                {
                    // The last send boundaries is either a blocking read of blocking MPI_Send,
                    // so we know everything in the queue is done
                    timers.sweep_time += wtime() - sweep_tick;
                }


                // Compute the Scalar Flux
                compute_scalar_flux(&problem, &rankinfo, &buffers, &events);
                if (problem.cmom-1 > 0)
                    compute_scalar_flux_moments(&problem, &rankinfo, &buffers, &events);

                // Get the new scalar flux back and check inner convergence
                copy_back_scalar_flux(&problem, &rankinfo, &buffers, memory.scalar_flux);

                double conv_tick = wtime();

                innerdone = inner_convergence(&problem, &rankinfo, &memory);

                if (profiling && rankinfo.rank == 0)
                    timers.convergence_time += wtime() - conv_tick;

                // Do any profiler updates for timings
                if (rankinfo.rank == 0)
                    inner_profiler(&timers, &problem, &events);

                if (innerdone)
                {
                    i += 1;
                    break;
                }

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
                outer_profiler(&timers, &events);

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

    cudaDeviceSynchronize();
    check_cuda("Finishing queue before simulation end");

    if (rankinfo.rank == 0)
    {
        timers.simulation_time = wtime() - timers.simulation_time;

        print_timing_report(&timers, &problem, total_iterations);
    }

    destroy_events(&events);

    free_memory(&memory);

    finish_comms();

    return EXIT_SUCCESS;
}

void print_banner(void)
{
    printf("\n");
    printf(" SNAP: SN (Discrete Ordinates) Application Proxy\n");
    printf(" MPI+CUDA port\n");
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

void print_cuda_info(void)
{

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    check_cuda("Getting device properties");

    printf("\n%s\n", STARS);
    printf("  CUDA Information\n");
    printf("%s\n", STARS);

    // Print out device name
    printf(" Device\n");
    printf("   %s\n", prop.name);
    printf("\n");

    // Driver version
    int driver, runtime;
    cudaDriverGetVersion(&driver);
    check_cuda("Getting driver version");
    cudaRuntimeGetVersion(&runtime);
    check_cuda("Getting runtime version");
    printf(" Driver\n");
    printf("   Driver %d\n", driver);
    printf("   Runtime %d\n", runtime);
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
        printf("   %-28s %6.3lfs\n", "MPI Send time", sweep_mpi_time);
        printf("   %-28s %6.3lfs\n", "MPI Recv time", sweep_mpi_recv_time);
        printf("   %-28s %6.3lfs\n", "PCIe transfer time", timers->sweep_transfer_time);
        printf("   %-28s %6.3lfs\n", "Compute time", timers->sweep_time-sweep_mpi_time-sweep_mpi_recv_time-timers->sweep_transfer_time);
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

