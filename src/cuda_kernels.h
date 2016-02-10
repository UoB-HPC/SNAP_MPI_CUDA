
#pragma once

/** \file
* \brief Contains function declarations of kernels
*/

#include "sweep.h"

__global__
void calc_dd_coeff(
    const double dx,
    const double dy,
    const double dz,
    const double * __restrict__ eta,
    const double * __restrict__ xi,
    double * __restrict__ dd_i,
    double * __restrict__ dd_j,
    double * __restrict__ dd_k
);

__global__
void calc_denominator(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,
    const double * __restrict__ mat_cross_section,
    const double * __restrict__ velocity_delta,
    const double * __restrict__ mu,
    const double * __restrict__ dd_i,
    const double * __restrict__ dd_j,
    const double * __restrict__ dd_k,
    double * __restrict__ denominator
);

__global__ 
void calc_velocity_delta(
    const double * __restrict__ velocities,
    const double dt,
    double * __restrict__ velocity_delta
);

__global__
void calc_inner_source(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int ng,
    const unsigned int cmom,
    const unsigned int nmom,
    const double * __restrict__ outer_source,
    const double * __restrict__ scattering_matrix,
    const double * __restrict__ scalar_flux,
    const double * __restrict__ scalar_flux_moments,
    double * __restrict__ inner_source
);

__global__
void calc_outer_source(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int ng,
    const unsigned int cmom,
    const unsigned int nmom,
    const double * __restrict__ fixed_source,
    const double * __restrict__ scattering_matrix,
    const double * __restrict__ scalar_flux,
    const double * __restrict__ scalar_flux_moments,
    double * __restrict__ outer_source
);


__global__
void reduce_flux(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,

    const double * __restrict__ angular_flux_in_0,
    const double * __restrict__ angular_flux_in_1,
    const double * __restrict__ angular_flux_in_2,
    const double * __restrict__ angular_flux_in_3,
    const double * __restrict__ angular_flux_in_4,
    const double * __restrict__ angular_flux_in_5,
    const double * __restrict__ angular_flux_in_6,
    const double * __restrict__ angular_flux_in_7,

    const double * __restrict__ angular_flux_out_0,
    const double * __restrict__ angular_flux_out_1,
    const double * __restrict__ angular_flux_out_2,
    const double * __restrict__ angular_flux_out_3,
    const double * __restrict__ angular_flux_out_4,
    const double * __restrict__ angular_flux_out_5,
    const double * __restrict__ angular_flux_out_6,
    const double * __restrict__ angular_flux_out_7,

    const double * __restrict__ velocity_delta,
    const double * __restrict__ quad_weights,

    double * __restrict__ scalar_flux
);


__global__
void reduce_flux_moments(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,
    const unsigned int cmom,

    const double * __restrict__ angular_flux_in_0,
    const double * __restrict__ angular_flux_in_1,
    const double * __restrict__ angular_flux_in_2,
    const double * __restrict__ angular_flux_in_3,
    const double * __restrict__ angular_flux_in_4,
    const double * __restrict__ angular_flux_in_5,
    const double * __restrict__ angular_flux_in_6,
    const double * __restrict__ angular_flux_in_7,

    const double * __restrict__ angular_flux_out_0,
    const double * __restrict__ angular_flux_out_1,
    const double * __restrict__ angular_flux_out_2,
    const double * __restrict__ angular_flux_out_3,
    const double * __restrict__ angular_flux_out_4,
    const double * __restrict__ angular_flux_out_5,
    const double * __restrict__ angular_flux_out_6,
    const double * __restrict__ angular_flux_out_7,

    const double * __restrict__ velocity_delta,
    const double * __restrict__ quad_weights,
    const double * __restrict__ scat_coeff,

    double * __restrict__ scalar_flux_moments
);

__global__
void sweep_plane_kernel(
    const unsigned int nx,
    const unsigned int ny,
    const unsigned int nz,
    const unsigned int nang,
    const unsigned int ng,
    const unsigned int cmom,
    const int istep,
    const int jstep,
    const int kstep,
    const unsigned int oct,
    const unsigned int z_pos,
    const unsigned int num_planes,
    const struct cell_id * plane,
    const double * __restrict__ source,
    const double * __restrict__ scat_coeff,
    const double * __restrict__ dd_i,
    const double * __restrict__ dd_j,
    const double * __restrict__ dd_k,
    const double * __restrict__ mu,
    const double * __restrict__ velocity_delta,
    const double * __restrict__ mat_cross_section,
    const double * __restrict__ angular_flux_in,
    double * __restrict__ flux_i,
    double * __restrict__ flux_j,
    double * __restrict__ flux_k,
    double * __restrict__ angular_flux_out
);

