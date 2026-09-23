#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "mpi_base.h"
#include "linalg.h"
#include "pb_stencil.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"                                                           
#else
#define EXTERN_C
#endif

/*
Apply a 3-D Laplace filter to a 3-D array with cyclic boundary conditions
The code uses an input array of shape (n+2, n, n) and output array of shape (n, n, n)
The +2 is used to either use memcpy to swap the top and bottom slices (skipping the % in the first loop)
or uses MPI to exchange the top and bottom slices between processes
@param u: the input array
@param u_new: the output array
@param n: the size of the array in each dimension
*/
void laplace_filter(double *u, double *u_new, int size1, int size2) {
    long int i, j, k;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k1, k2;
    long int n2 = size2 * size2;

    if (u == u_new) {
        mpi_fprintf(stderr, "laplace_filter: u and u_new are the same array (in-place operation not supported)\n");
        exit(1);
    }

    // Precompute neighbor indices for periodic BCs in j and k
    int jprev[size2];
    int jnext[size2];
    int kprev[size2];
    int knext[size2];
    for (int t = 0; t < size2; ++t) {
        kprev[t] = ((t - 1 + size2 ) % size2);
        knext[t] = ((t + 1      ) % size2);
        jprev[t] = kprev[t] * size2;
        jnext[t] = knext[t] * size2;
    }

    // Exchange the top and bottom slices
    mpi_grid_exchange_bot_top(u, size1, size2);

    #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k1, k2)
    for (i = 0; i < size1; i++) {
        i0 = i * n2;
        i1 = i0 + n2;
        i2 = i0 - n2;
        for (j = 0; j < size2; j++) {
            j0 = j * size2;
            j1 = jnext[j];
            j2 = jprev[j];
            for (k = 0; k < size2; k++) {
                k1 = knext[k];
                k2 = kprev[k];
                u_new[i0 + j0 + k] = (
                    u[i1 + j0 + k] +
                    u[i2 + j0 + k] +
                    u[i0 + j1 + k] +
                    u[i0 + j2 + k] +
                    u[i0 + j0 + k1] +
                    u[i0 + j0 + k2] -
                    u[i0 + j0 + k] * 6.0
                    );
            }
        }
    }
}

EXTERN_C void laplace_filter_pb(
    double *u, double *u_new, int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
    long int i, j, k;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k1, k2;
    long int n2 = size2 * size2;

    long int idx0, idx_x, idx_y, idx_z;

    if (u == u_new) {
        mpi_fprintf(stderr, "laplace_filter_pb: u and u_new are the same array (in-place operation not supported)\n");
        exit(1);
    }

    // Precompute neighbor indices for periodic BCs in j and k
    int jprev[size2];
    int jnext[size2];
    int kprev[size2];
    int knext[size2];
    for (int t = 0; t < size2; ++t) {
        kprev[t] = ((t - 1 + size2 ) % size2);
        knext[t] = ((t + 1      ) % size2);
        jprev[t] = kprev[t] * size2;
        jnext[t] = knext[t] * size2;
    }

    // Exchange the top and bottom slices
    mpi_grid_exchange_bot_top(u, size1, size2);
    mpi_grid_exchange_bot_top(eps_x, size1, size2);
    mpi_grid_exchange_bot_top(eps_y, size1, size2);
    mpi_grid_exchange_bot_top(eps_z, size1, size2);

    #pragma omp parallel for if(size1 * n2 >= 40000)
    for (i = 0; i < size1; i++) {
        const long ii0 = i * n2;
        for (j = 0; j < size2; j++) {
            const long c = ii0 + j * size2;
            pb_stencil_row(
                u, eps_x, eps_y, eps_z, k2_screen,
                c, c + n2, c - n2, ii0 + jnext[j], ii0 + jprev[j], size2, u_new + c
            );
        }
    }
}

/*
 * Fused PB residual: same stencil as laplace_filter_pb, but subtracts rhs in the
 * same pass. If out != NULL it stores out = rhs - A u; if norm != NULL it returns
 * max|A u - rhs| (over all MPI ranks) in *norm. Values are identical to
 * laplace_filter_pb + dscal + daxpy (or + daxpy + norm_inf).
 */
EXTERN_C void laplace_filter_pb_residual(
    double *u, double *rhs, double *out, double *norm, int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
    const long int n2 = (long int)size2 * size2;
    double max_val = 0.0;

    if (u == out) {
        mpi_fprintf(stderr, "laplace_filter_pb_residual: u and out are the same array (in-place operation not supported)\n");
        exit(1);
    }

    int jprev[size2];
    int jnext[size2];
    int kprev[size2];
    int knext[size2];
    for (int t = 0; t < size2; ++t) {
        kprev[t] = ((t - 1 + size2 ) % size2);
        knext[t] = ((t + 1      ) % size2);
        jprev[t] = kprev[t] * size2;
        jnext[t] = knext[t] * size2;
    }

    mpi_grid_exchange_bot_top(u, size1, size2);
    mpi_grid_exchange_bot_top(eps_x, size1, size2);
    mpi_grid_exchange_bot_top(eps_y, size1, size2);
    mpi_grid_exchange_bot_top(eps_z, size1, size2);

    #pragma omp parallel for if((long int)size1 * n2 >= 40000) reduction(max:max_val)
    for (long int i = 0; i < size1; i++) {
        const long int i0 = i * n2;
        double row[size2];
        for (long int j = 0; j < size2; j++) {
            const long c = i0 + j * size2;
            pb_stencil_row(
                u, eps_x, eps_y, eps_z, k2_screen,
                c, c + n2, c - n2, i0 + jnext[j], i0 + jprev[j], size2, row
            );
            for (long int k = 0; k < size2; k++) {
                double d = row[k] - rhs[c + k];
                if (out != NULL) out[c + k] = -d;
                if (norm != NULL) {
                    double f = fabs(d);
                    if (f > max_val) max_val = f;
                }
            }
        }
    }

    if (norm != NULL) {
        allreduce_max(&max_val, 1);
        *norm = max_val;
    }
}
