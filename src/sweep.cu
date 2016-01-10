
#include "sweep.h"

void init_planes(struct plane** planes, unsigned int *num_planes, struct problem * problem, struct rankinfo * rankinfo)
{
    *num_planes = rankinfo->nx + rankinfo->ny + problem->chunk - 2;
    *planes = (struct plane *)malloc(sizeof(struct plane) * *num_planes);

    for (unsigned int p = 0; p < *num_planes; p++)
        (*planes)[p].num_cells = 0;

    for (unsigned int k = 0; k < problem->chunk; k++)
        for (unsigned int j = 0; j < rankinfo->ny; j++)
            for (unsigned int i = 0; i < rankinfo->nx; i++)
            {
                unsigned int p = i + j + k;
                (*planes)[p].num_cells += 1;
            }

    for (unsigned int p = 0; p < *num_planes; p++)
    {
        (*planes)[p].cell_ids = (struct cell_id *)malloc(sizeof(struct cell_id) * (*planes)[p].num_cells);
    }

    unsigned int index[*num_planes];
    for (unsigned int p = 0; p < *num_planes; p++)
        index[p] = 0;

    for (unsigned int k = 0; k < problem->chunk; k++)
        for (unsigned int j = 0; j < rankinfo->ny; j++)
            for (unsigned int i = 0; i < rankinfo->nx; i++)
            {
                unsigned int p = i + j + k;
                (*planes)[p].cell_ids[index[p]].i = i;
                (*planes)[p].cell_ids[index[p]].j = j;
                (*planes)[p].cell_ids[index[p]].k = k;
                index[p] += 1;
            }
}

void copy_planes(const struct plane * planes, const unsigned int num_planes, struct buffers * buffers)
{
    buffers->planes = (struct cell_id **)malloc(sizeof(struct cell_id *)*num_planes);

    for (unsigned int p = 0; p < num_planes; p++)
    {
        cudaMalloc(&(buffers->planes[p]),
            sizeof(struct cell_id)*planes[p].num_cells);
        check_cuda("Creating a plane cell indicies buffer");
        cudaMemcpy(buffers->planes[p], planes[p].cell_ids,
            sizeof(struct cell_id)*planes[p].num_cells, cudaMemcpyHostToDevice);
        check_cuda("Creating and copying a plane cell indicies buffer");
    }
}

void sweep_plane(
    const unsigned int z_pos,
    const int octant,
    const int istep,
    const int jstep,
    const int kstep,
    const unsigned int plane,
    const struct plane * planes,
    struct problem * problem,
    struct rankinfo * rankinfo,
    struct buffers * buffers
    )
{

    // 2 dimensional kernel
    // First dimension: number of angles * number of groups
    // Second dimension: number of cells in plane
    dim3 blocks(problem->nang*problem->ng, planes[plane].num_cells, 1);
    dim3 threads(1, 1, 1);

    sweep_plane_kernel<<< blocks, threads >>>(
        rankinfo->nx, rankinfo->ny, rankinfo->nz,
        problem->nang, problem->ng, problem->cmom,
        istep, jstep, kstep, octant, z_pos,
        buffers->planes[plane], buffers->inner_source,
        buffers->scat_coeff,
        buffers->dd_i, buffers->dd_j, buffers->dd_k,
        buffers->mu, buffers->velocity_delta,
        buffers->mat_cross_section, buffers->denominator,
        buffers->angular_flux_in[octant],
        buffers->flux_i, buffers->flux_j, buffers->flux_k,
        buffers->angular_flux_out[octant]
    );
    check_cuda("Enqueue plane sweep kernel");
}
