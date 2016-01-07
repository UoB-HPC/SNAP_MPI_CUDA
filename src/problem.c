
#include "problem.h"
#include <math.h>

void init_quadrature_weights(
    const struct problem * problem,
    const struct buffers * buffers
    )
{
    // Create tempoary on host for quadrature weights
    double *quad_weights = malloc(sizeof(double)*problem->nang);
    // Uniform weights
    for (unsigned int a = 0; a < problem->nang; a++)
    {
        quad_weights[a] = 0.125 / (double)(problem->nang);
    }

    // Copy to device
    cudaMemcpy(buffers->quad_weights, quad_weights,
        sizeof(double)*problem->nang,
        cudaMemcpyHostToDevice);
    check_cuda("Copying quadrature weights to device");
    free(quad_weights);
}

void calculate_cosine_coefficients(const struct problem * problem,
    const struct buffers * buffers,
    double * restrict mu,
    double * restrict eta,
    double * restrict xi
    )
{

    double dm = 1.0 / problem->nang;

    mu[0] = 0.5 * dm;
    eta[0] = 1.0 - 0.5 * dm;
    double t = mu[0] * mu[0] + eta[0] * eta[0];
    xi[0] = sqrt(1.0 - t);

    for (unsigned int a = 1; a < problem->nang; a++)
    {
        mu[a] = mu[a-1] + dm;
        eta[a] = eta[a-1] - dm;
        t = mu[a] * mu[a] + eta[a] * eta[a];
        xi[a] = sqrt(1.0 - t);
    }

    // Copy to device
    cudaMemcpy(buffers->mu, mu, sizeof(double)*problem->nang, cudaMemcpyHostToDevice);
    check_cuda("Copying mu cosine to device");
    cudaMemcpy(buffers->eta, eta, sizeof(double)*problem->nang, cudaMemcpyHostToDevice);
    check_cuda("Copying eta cosine to device");
    cudaMemcpy(buffers->xi, xi, sizeof(double)*problem->nang, cudaMemcpyHostToDevice);
    check_cuda("Copying xi cosine to device");

}

void calculate_scattering_coefficients(
    const struct problem * problem,
    const struct buffers * buffers,
    const double * restrict mu,
    const double * restrict eta,
    const double * restrict xi
    )
{
    // Allocate temporary on host for scattering coefficients
    double *scat_coeff = malloc(sizeof(double)*problem->nang*problem->cmom*8);
    // (mu*eta*xi)^l starting at 0
    for (int id = 0; id < 2; id++)
    {
        double is = (id == 1) ? 1.0 : -1.0;
        for (int jd = 0; jd < 2; jd++)
        {
            double js = (jd == 1) ? 1.0 : -1.0;
            for (int kd = 0; kd < 2; kd++)
            {
                double ks = (kd == 1) ? 1.0 : -1.0;
                int oct = 4*id + 2*jd + kd;
                // Init first moment
                for (unsigned int a = 0; a < problem->nang; a++)
                    scat_coeff[SCAT_COEFF_INDEX(a,0,oct,problem->nang,problem->cmom)] = 1.0;
                // Init other moments
                int mom = 1;
                for (int l = 1; l < problem->nmom; l++)
                {
                    for (int m = 0; m < 2*l+1; m++)
                    {
                        for (unsigned int a = 0; a < problem->nang; a++)
                        {
                            scat_coeff[SCAT_COEFF_INDEX(a,mom,oct,problem->nang,problem->cmom)] = pow(is*mu[a], 2.0*l-1.0) * pow(ks*xi[a]*js*eta[a], m);
                        }
                        mom += 1;
                    }
                }
            }
        }
    }

    // Copy to device
    cudaMemcpy(buffers->scat_coeff, scat_coeff, 
        sizeof(double)*problem->nang*problem->cmom*8,
        cudaMemcpyHostToDevice);
    check_cuda("Copying scattering coefficients to device");
    free(scat_coeff);
}

void init_material_data(
    const struct problem * problem,
    const struct buffers * buffers,
    double * restrict mat_cross_section
    )
{
    mat_cross_section[0] = 1.0;
    for (unsigned int g = 1; g < problem->ng; g++)
    {
        mat_cross_section[g] = mat_cross_section[g-1] + 0.01;
    }
    // Copy to device
    cudaMemcpy(buffers->mat_cross_section, mat_cross_section,
        sizeof(double)*problem->ng, cudaMemcpyHostToDevice);
    check_cuda("Copying material cross sections to device");
}

void init_fixed_source(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    const struct buffers * buffers
    )
{
    // Allocate temporary array for fixed source
    double *fixed_source = malloc(sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz);

    // Source everywhere, set at strength 1.0
    // This is src_opt == 0 in original SNAP
    for(unsigned int k = 0; k < rankinfo->nz; k++)
        for(unsigned int j = 0; j < rankinfo->ny; j++)
            for(unsigned int i = 0; i < rankinfo->nx; i++)
                for(unsigned int g = 0; g < problem->ng; g++)
                    fixed_source[FIXED_SOURCE_INDEX(g,i,j,k,problem->ng,rankinfo->nx,rankinfo->ny)] = 1.0;

    // Copy to device
    cudaMemcpy(buffers->fixed_source, fixed_source,
        sizeof(double)*problem->ng*rankinfo->nx*rankinfo->ny*rankinfo->nz, cudaMemcpyHostToDevice);
    check_cuda("Copying fixed source to device");
    free(fixed_source);
}

void init_scattering_matrix(
    const struct problem * problem,
    const struct buffers * buffers,
    const double * restrict mat_cross_section
    )
{
    // Allocate temporary array for scattering matrix
    double *scattering_matrix = malloc(sizeof(double)*problem->nmom*problem->ng*problem->ng);

    // 10% up scattering
    // 20% in group scattering
    // 70% down scattering
    // First and last group, no up/down scattering
    for (unsigned int g = 0; g < problem->ng; g++)
    {
        if (problem->ng == 1)
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,0,0,problem->nmom,problem->ng)] = mat_cross_section[g] * 0.5;
            break;
        }

        scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g,problem->nmom,problem->ng)] = 0.2 * 0.5 * mat_cross_section[g];

        if (g > 0)
        {
            double t = 1.0 / (double)(g);
            for (unsigned int g2 = 0; g2 < g; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g2,problem->nmom,problem->ng)] = 0.1 * 0.5 * mat_cross_section[g] * t;
            }
        }
        else
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,0,0,problem->nmom,problem->ng)] = 0.3 * 0.5 * mat_cross_section[0];
        }

        if (g < (problem->ng) - 1)
        {
            double t = 1.0 / (double)(problem->ng - g - 1);
            for (unsigned int g2 = g + 1; g2 < problem->ng; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(0,g,g2,problem->nmom,problem->ng)] = 0.7 * 0.5 * mat_cross_section[g] * t;
            }
        }
        else
        {
            scattering_matrix[SCATTERING_MATRIX_INDEX(0,problem->ng-1,problem->ng-1,problem->nmom,problem->ng)] = 0.9 * 0.5 * mat_cross_section[problem->ng-1];
        }
    }

    // Set scattering moments (up to 4)
    // Second moment 10% of first, subsequent half of previous
    if (problem->nmom > 1)
    {
        for (unsigned int g1 = 0; g1 < problem->ng; g1++)
        {
            for (unsigned int g2 = 0; g2 < problem->ng; g2++)
            {
                scattering_matrix[SCATTERING_MATRIX_INDEX(1,g1,g2,problem->nmom,problem->ng)] = 0.1 * scattering_matrix[SCATTERING_MATRIX_INDEX(0,g1,g2,problem->nmom,problem->ng)];
                for (unsigned int m = 2; m < problem->nmom; m++)
                {
                    scattering_matrix[SCATTERING_MATRIX_INDEX(m,g1,g2,problem->nmom,problem->ng)] = 0.5 * scattering_matrix[SCATTERING_MATRIX_INDEX(m-1,g1,g2,problem->nmom,problem->ng)];
                }
            }
        }
    }

    // Copy to device
    cudaMemcpy(buffers->scattering_matrix, scattering_matrix,
        sizeof(double)*problem->nmom*problem->ng*problem->ng, cudaMemcpyHostToDevice);
    check_cuda("Copying scattering matrix to device");
    free(scattering_matrix);
}

void init_velocities(
    const struct problem * problem,
    const struct buffers * buffers
    )
{
    // Allocate tempoary array for velocities
    double *velocities = malloc(sizeof(double)*problem->ng);

    for (unsigned int g = 0; g < problem->ng; g++)
        velocities[g] = (double)(problem->ng - g);

    // Copy to device
    cudaMemcpy(buffers->velocities, velocities,
        sizeof(double)*problem->ng, cudaMemcpyHostToDevice);
    check_cuda("Copying velocities to device");
    free(velocities);
}

void init_velocity_delta(
    const struct problem * problem,
    const struct buffers * buffers
    )
{
    // We do this on the device because SNAP does it every outer
    calc_velocity_delta<<< problem->ng, 1 >>>(
        buffers->velocities, problem->dt, buffers->velocity_delta);
    check_cuda("Enqueue velocity delta calculation kernel");
}

void calculate_dd_coefficients(
    const struct problem * problem,
    const struct buffers * buffers
    )
{
    // We do this on the device because SNAP does it every outer
    calc_dd_coeff<<< problem->nang, 1 >>>(
        problem->dx, problem->dy, problem->dz,
        buffers->eta, buffers->xi,
        buffers->dd_i, buffers->dd_j, buffers->dd_k
    );
    check_cuda("Enqueue diamond different calculation kernel");
}

void calculate_denominator(
    const struct problem * problem,
    const struct rankinfo * rankinfo,
    const struct context * context,
    const struct buffers * buffers
    )
{
    // We do this on the device because SNAP does it every outer
    cl_int err;
    err = clSetKernelArg(context->kernels.calc_denominator, 0, sizeof(unsigned int), &rankinfo->nx);
    err |= clSetKernelArg(context->kernels.calc_denominator, 1, sizeof(unsigned int), &rankinfo->ny);
    err |= clSetKernelArg(context->kernels.calc_denominator, 2, sizeof(unsigned int), &rankinfo->nz);
    err |= clSetKernelArg(context->kernels.calc_denominator, 3, sizeof(unsigned int), &problem->nang);
    err |= clSetKernelArg(context->kernels.calc_denominator, 4, sizeof(unsigned int), &problem->ng);
    err |= clSetKernelArg(context->kernels.calc_denominator, 5, sizeof(cl_mem), &buffers->mat_cross_section);
    err |= clSetKernelArg(context->kernels.calc_denominator, 6, sizeof(cl_mem), &buffers->velocity_delta);
    err |= clSetKernelArg(context->kernels.calc_denominator, 7, sizeof(cl_mem), &buffers->mu);
    err |= clSetKernelArg(context->kernels.calc_denominator, 8, sizeof(cl_mem), &buffers->dd_i);
    err |= clSetKernelArg(context->kernels.calc_denominator, 9, sizeof(cl_mem), &buffers->dd_j);
    err |= clSetKernelArg(context->kernels.calc_denominator, 10, sizeof(cl_mem), &buffers->dd_k);
    err |= clSetKernelArg(context->kernels.calc_denominator, 11, sizeof(cl_mem), &buffers->denominator);
    check_ocl(err, "Setting denominator kernel arguments");

    size_t global[] = {problem->nang, problem->ng};
    err = clEnqueueNDRangeKernel(context->queue,
        context->kernels.calc_denominator,
        2, 0, global, NULL,
        0, NULL, &denominator_event);
    check_ocl(err, "Enqueue denominator kernel");
}
