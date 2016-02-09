
#pragma once

/** \file
* \brief Scalar flux reduction routines
*/

#define RED_BLOCK_SIZE 4

#include <math.h>

#include "global.h"

#include "cuda_global.h"
#include "cuda_buffers.h"

#include "profiler.h"

#ifdef __cplusplus
extern "C"
{
#endif

/** \brief Enqueue kernel to compute scalar flux (non-blocking) */
void compute_scalar_flux(struct problem * problem, struct rankinfo * rankinfo, struct buffers * buffers, struct events * events);

/** \brief Enqueue kernel to compute scalar flux moments (non-blocking) */
void compute_scalar_flux_moments(struct problem * problem, struct rankinfo * rankinfo, struct buffers * buffers, struct events * events);


/** \brief Copy the scalar flux back to the host (choose blocking) */
void copy_back_scalar_flux(struct problem *problem, struct rankinfo * rankinfo, struct buffers * buffers, double * scalar_flux);

#ifdef __cplusplus
}
#endif

