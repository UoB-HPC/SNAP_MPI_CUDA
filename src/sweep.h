
#pragma once

/** \file
* \brief Sweep calculation routines
*/

#include <stdlib.h>

#include "global.h"
#include "cuda_global.h"
#include "cuda_buffers.h"

/** \brief Structure to hold a 3D cell index for use in storing planes */
struct cell_id
{
    /** @{ \brief Cell index */
    unsigned int i, j, k;
    /** @} */
};

/** \brief Structure to hold list of cells in each plane */
struct plane
{
    /** \brief Number of cells in this plane */
    unsigned int num_cells;
    /** \brief Array of cell indexes in this plane */
    struct cell_id * cell_ids;
};

#ifdef __cplusplus
extern "C"
{
#endif

/** \brief Create a list of cell indexes in the planes in the XY plane determined by chunk */
void init_planes(struct plane** planes, unsigned int *num_planes, struct problem * problem, struct rankinfo * rankinfo);

/** \brief Copy the array of planes to the OpenCL buffers */
void copy_planes(const struct plane * planes, const unsigned int num_planes, struct buffers * buffers);

/** \brief Enqueue the kernels to sweep a plane */
void sweep_plane(const unsigned int z_pos, const int octant, const int istep, const int jstep, const int kstep, const unsigned int plane, const struct plane * planes, struct problem * problem, struct rankinfo * rankinfo, struct buffers * buffers);

#ifdef __cplusplus
}
#endif

