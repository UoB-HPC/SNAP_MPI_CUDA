
/** \file
* \brief Manage the allocation of OpenCL buffers
*/

#pragma once

#include "cuda_global.h"
#include "global.h"

/** \brief Struct to contain all the OpenCL buffers */
struct buffers
{
    /** @{ \brief
    Angular flux - two copies for time dependence, each ocant in own buffer
    */
    double * angular_flux_in[8];
    double * angular_flux_out[8];
    /** @} */

    /** @{
    \brief Edge flux arrays */
    double * flux_i, flux_j, flux_k;
    /** @} */

    /** @{ \brief Scalar flux arrays */
    double * scalar_flux;
    double * scalar_flux_moments;
    /** @} */

    /** \brief Quadrature weights */
    double * quad_weights;

    /** @{ \brief Cosine coefficients */
    double * mu, eta, xi;
    /** @} */

    /** \brief Scattering coefficient */
    double * scat_coeff;

    /** \brief Material cross section */
    double * mat_cross_section;

    /** @{ \brief Source terms */
    double * fixed_source;
    double * outer_source;
    double * inner_source;
    /** @} */

    /** \brief Scattering terms */
    double * scattering_matrix;

    /** @{ \brief Diamond diference co-efficients */
    double * dd_i, dd_j, dd_k;
    /** @} */

    /** \brief Mock velocities array */
    double * velocities;

    /** \brief Time absorption coefficient */
    double * velocity_delta;

    /** \brief Transport denominator */
    double * denominator;

    /** \brief Lists of cell indicies in each plane
    Each buffer is an array of the i,j,k indicies for cells within that plane
    One buffer per plane */
    double * *planes;
};

/** \brief Check the device has enough memory to allocate the buffers */
void check_device_memory_requirements(struct problem * problem, struct rankinfo * rankinfo, struct context * context);

/** \brief Allocate global device memory */
void allocate_buffers(struct problem * problem, struct rankinfo * rankinfo, struct buffers * buffers);

/** \brief Launch a kernel to zero the (1D) buffer (non-blocking) */
void zero_buffer(struct context * context, double * buffer, size_t offset, size_t size);

/** \brief Swap the angular flux pointers around (in <-> out) */
void swap_angular_flux_buffers(struct buffers * buffers);

