#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "verlet.h"
#include "constants.h"
#include "mp_structs.h"
#include "mpi_base.h"
#include "linalg.h"
#include "mp_structs.h"

#define EPS_FIELD_TOL 1e-1
#define EPS_FIELD_MAX_ITER 500

#ifdef __cplusplus
#define EXTERN_C extern "C"                                                           
#else
#define EXTERN_C
#endif

/*
Compute the provisional update for the field phi using the Verlet algorithm.
- phi(t) = 2 * phi(t-1) - phi(t-2)
@param phi: the potential field at the current time step
@param phi_prev: the potential field at the previous time step
@param size: the total number of grid points
*/
void verlet_update(double *phi, double *phi_prev, long int size) {
    double app;
    #pragma omp parallel for private(app)
    for (long int i = 0; i < size; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }
}

// Warm-start coefficients: base, Verlet-like linear, and second-order extrapolation.
static const double POLY_COEF[MAZE_Y_HIST_MAX + 1][MAZE_Y_HIST_MAX + 1] = {
    {1.0},
    {2.0, -1.0},
    {3.0, -3.0, 1.0},
};

static void fill_extrapolation_coefficients(y_extrap_order y_extrap, int y_hist_len, double *coef, int *order) {
    for (int i = 0; i <= MAZE_Y_HIST_MAX; i++) {
        coef[i] = 0.0;
    }

    int effective_order = y_extrap;
    if (effective_order > y_hist_len) {
        effective_order = y_hist_len;
    }

    for (int i = 0; i <= effective_order; i++) {
        coef[i] = POLY_COEF[effective_order][i];
    }
    *order = effective_order;
}

// Evaluate an order-p predictor using y^{k-1} and older history terms.
static void extrap_predict(double *out, double *y_km1, double **y_hist, const double *c, int p, long int size) {
    vec_copy(y_km1, out, size);
    dscal(out, c[0], size);
    for (int j = 1; j <= p; j++) {
        daxpy(y_hist[j - 1], out, c[j], size);
    }
}

// Snapshot y^{k-1} and overwrite y with the selected warm-start predictor.
static void y_build_guess(double *y, double **y_hist, y_extrap_order y_extrap, int y_hist_len, long int size) {
    double *y_km1 = y_hist[MAZE_Y_HIST_MAX];  // spare slot -> snapshot of y^{k-1}
    double coef[MAZE_Y_HIST_MAX + 1];
    int order;
    vec_copy(y, y_km1, size);

    if (y_extrap == MAZE_Y_GUESS_ZERO) {
        memset(y, 0, size * sizeof(double));
        return;
    }

    fill_extrapolation_coefficients(y_extrap, y_hist_len, coef, &order);
    extrap_predict(y, y_km1, y_hist, coef, order, size);
}

static void y_shift_history(double **y_hist, int *y_hist_len) {
    double *newest = y_hist[MAZE_Y_HIST_MAX];
    for (int i = MAZE_Y_HIST_MAX; i > 0; i--) {
        y_hist[i] = y_hist[i - 1];
    }
    y_hist[0] = newest;

    if (*y_hist_len < MAZE_Y_HIST_MAX) {
        (*y_hist_len)++;
    }
}

/*
Apply Verlet algorithm to compute the updated value of the field phi, with LCG + SHAKE.
The previous and current fields and the y array are updated in place.
@param tol: tolerance
@param h: the grid spacing
@param phi: the potential field of size n_grid * n_grid * n_grid
@param phi_prev: electrostatic field for step t - 1 Verlet
@param q: the charge on a grid of size n_grid * n_grid * n_grid\
@param y: copy of the 'q' given as input to the function
@param n_grid: the number of grid points in each dimension
@param precond: the preconditioner function

@return the number of iterations for convergence of the LCG
*/
EXTERN_C int verlet_poisson(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2,
    void (*precond)(double *, double *, int, int, int)
) {
    int iter_conv;
    long int n3 = size1 * size2 * size2;

    double *tmp = (double*)malloc(n3 * sizeof(double));
    
    // Compute provisional update for the field phi
    verlet_update(phi, phi_prev, n3);

    // Compute the constraint with the provisional value of the field phi
    laplace_filter(phi, tmp, size1, size2);
    daxpy(q, tmp, (4 * M_PI) / h, n3);  // sigma_p = A . phi + 4 * pi * rho / eps

    // Apply LCG
    if (precond == NULL) {
        iter_conv = conj_grad(tmp, y, y, tol, size1, size2);  // Inplace y <- y0 - tolerance scaled by 4*pi/h to guarantee correct sigma_p = A . phi . h/4pi + rho/eps 
    } else {
        iter_conv = conj_grad_precond(tmp, y, y, tol, size1, size2, precond);  // Inplace y <- y0 - tolerance scaled by 4*pi/h to guarantee correct sigma_p = A . phi . h/4pi + rho/eps 
    }

    // Scale the field with the constrained 'force' term
    daxpy(y, phi, -1.0, n3);  // phi = phi - y

    // Free temporary arrays
    free(tmp);

    return iter_conv;
}

/*
Apply Verlet algorithm to compute the updated value of the field phi, with Multigrid + SHAKE.
The previous and current fields and the y array are updated in place.
@param tol: tolerance
@param h: the grid spacing
@param phi: the potential field of size n_grid * n_grid * n_grid
@param phi_prev: electrostatic field for step t - 1 Verlet
@param q: the charge on a grid of size n_grid * n_grid * n_grid\
@param y: copy of the 'q' given as input to the function
@param n_grid: the number of grid points in each dimension

@return the number of iterations for convergence of the LCG
*/
EXTERN_C int verlet_poisson_multigrid(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    double** y_hist, y_extrap_order y_extrap,
    int *y_hist_len,
    int size1, int size2
) {
    int res = -1;

    long int n3 = size1 * size2 * size2;

    double constant;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    double *q_rhs = mpi_grid_allocate(size1, size2);

    // Compute provisional update for the field phi
    verlet_update(phi, phi_prev, n3);

    constant = (4 * M_PI) / h;
    laplace_filter(phi, tmp, size1, size2);
    laplace_filter_rhs(q, q_rhs, size1, size2);
    daxpy(q_rhs, tmp, constant, n3);  // sigma_p = A . phi + 4 * pi * B.rho / eps
    // memset(y, 0, n3 * sizeof(double));
    // printf("\nprima y = %e\n", norm_inf(y, n3));

    // Build the y_0 initial guess for the multigrid solve.
    y_build_guess(y, y_hist, y_extrap, *y_hist_len, n3);

    res = multigrid_solve(tol, tmp, y, size1, size2, get_n_start());

    // Keep y^{k-1} available for the next warm-start predictor.
    y_shift_history(y_hist, y_hist_len);

    // Scale the field with the constrained 'force' term
    daxpy(y, phi, -1.0, n3);  // phi = phi - y

    // Free temporary arrays
    free(tmp);
    mpi_grid_free(q_rhs, size2);

    if (res == -1) {
        fprintf(stderr, "Warning: Multigrid did not converge after 1000 iterations.\n");    
    }

    return res;
}


EXTERN_C int verlet_poisson_pb(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
    int iter_conv;
    long int n3 = size1 * size2 * size2;

    double *tmp = (double*)malloc(n3 * sizeof(double));
    
    // Compute provisional update for the field phi
    verlet_update(phi, phi_prev, n3);

    // Compute the constraint with the provisional value of the field phi
    laplace_filter_pb(
        phi, tmp, size1, size2,
        eps_x, eps_y, eps_z, k2_screen
    );
    daxpy(q, tmp, (4 * M_PI) / h, n3);  // sigma_p = A . phi + 4 * pi * rho

    // Apply LCG
    iter_conv = conj_grad_pb(
        tmp, y, y, tol, size1, size2,
        eps_x, eps_y, eps_z, k2_screen
    );  // Inplace y <- y0

    // Scale the field with the constrained 'force' term
    daxpy(y, phi, -1.0, n3);  // phi = phi - y

    // Free temporary arrays
    free(tmp);

    return iter_conv;
}


/*
Apply Verlet algorithm to compute the updated value of the field phi, with Multigrid + SHAKE.
The previous and current fields and the y array are updated in place.
@param tol: tolerance
@param h: the grid spacing
@param phi: the potential field of size n_grid * n_grid * n_grid
@param phi_prev: electrostatic field for step t - 1 Verlet
@param q: the charge on a grid of size n_grid * n_grid * n_grid\
@param y: copy of the 'q' given as input to the function
@param n_grid: the number of grid points in each dimension
@param eps_x, eps_y, eps_z: the spatially dependent dielectric constants in each direction
@param k2_screen: the spatially dependent screening term for the linearized PB equation
@return the number of iterations for convergence of the MG or -1 if MG did not converge
*/
EXTERN_C int verlet_pb_multigrid(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    double** y_hist, y_extrap_order y_extrap,
    int *y_hist_len,
    int size1, int size2, double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
    int res = -1;

    long int n3 = size1 * size2 * size2;

    double constant;
    double *tmp = (double*)malloc(n3 * sizeof(double));

    // Compute provisional update for the field phi
    verlet_update(phi, phi_prev, n3);

    constant = (4 * M_PI) / h;
    laplace_filter_pb(phi, tmp, size1, size2, eps_x, eps_y, eps_z, k2_screen);
    daxpy(q, tmp, constant, n3);  // sigma_p = A_pb . phi + 4 * pi * q / h = sigma_p^k

    // Build the y_0 initial guess for the multigrid solve.
    y_build_guess(y, y_hist, y_extrap, *y_hist_len, n3);

    res = multigrid_solve_pb(
        tol, tmp, y, size1, size2, get_n_start(),
        eps_x, eps_y, eps_z, k2_screen
    );  // solve A_pb . y = sigma_p

    // Keep y^{k-1} available for the next warm-start predictor.
    y_shift_history(y_hist, y_hist_len);

    // Scale the field with the constrained 'force' term
    daxpy(y, phi, -1.0, n3);  // phi = phi - y

    // Free temporary arrays
    free(tmp);

    if (res == -1) {
        fprintf(stderr, "Warning: Multigrid did not converge after 1000 iterations.\n");    
    }

    return res;
}

EXTERN_C int verlet_pb_multigrid_eps_field(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2, double *eps_x, double *eps_y, double *eps_z, double *k2_screen,
    grid *grid_ctx
) {
    // TODO ideally these function should be agnostic to the structure of the grid and particle structs
    if (grid_ctx == NULL || grid_ctx->kBT <= 0.0) {
        mpi_printf("Warning: field-dependent dielectric requested without valid kBT. Running single PB solve.\n");
    }
    int res = -1;
    int iter_conv = 0;

    long int i;
    long int n3 = (long int)size1 * size2 * size2;

    double app, constant;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    double *tmp2 = (double*)malloc(n3 * sizeof(double));

    // Provisional Verlet update
    #pragma omp parallel for private(app)
    for (i = 0; i < n3; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }

    constant = (4 * M_PI) / h;

    double max_diff = EPS_FIELD_TOL + 1.0;
    int eps_iter = 0;
    if (grid_ctx != NULL) {
        grid_ctx->eps_phi_iters = 0;
    }

    while (max_diff > EPS_FIELD_TOL && eps_iter < EPS_FIELD_MAX_ITER) {
        if (grid_ctx != NULL && grid_ctx->kBT > 0.0) {
            max_diff = grid_update_eps_field_dependent(grid_ctx, NULL); 
        } else {
            max_diff = 0.0;  // Skip epsilon loop if no context
        }

        // Recompute sigma_p = A_pb . phi + 4*pi*q/h with updated eps
        laplace_filter_pb(phi, tmp2, size1, size2, eps_x, eps_y, eps_z, k2_screen);
        daxpy(q, tmp2, constant, n3);

        res = -1;
        iter_conv = 0;
        while(iter_conv < MG_ITER_LIMIT_PB) { 
            multigrid_pb_apply(tmp2, y, size1, size2, get_n_start(), MG_SOLVE_SM_PB, eps_x, eps_y, eps_z, k2_screen); //solve A_pb . y = sigma_p

            laplace_filter_pb(y, tmp, size1, size2, eps_x, eps_y, eps_z, k2_screen);
            daxpy(tmp2, tmp, -1., n3);  // res = A . y - sigma_p
            app = norm_inf(tmp, n3);   // Compute norm_inf of residual
            iter_conv++;
            
            // printf("\ny = %e \t iter=%d \t res=%e\n", norm_inf(y, n3), iter_conv,app);
            
            if (app <= tol){
                res = iter_conv;
                break;
            }
        }
        daxpy(y, phi, -1.0, n3);  // phi = phi - y

        eps_iter++;
        // mpi_printf("Field-dependent dielectric iteration %d: max diff = %e\n", eps_iter, max_diff);

        /* Stop early if we were not actually iterating eps */
        if (grid_ctx == NULL || grid_ctx->kBT <= 0.0) {
            break;
        }
    }

    free(tmp);
    free(tmp2);

    if (max_diff > EPS_FIELD_TOL && grid_ctx != NULL && grid_ctx->kBT > 0.0) {
        mpi_printf("Warning: dielectric update did not converge after %d iterations (max diff = %e)\n", eps_iter, max_diff);
    }// else if (grid_ctx != NULL && grid_ctx->kBT > 0.0) {
        // mpi_printf(
        //     "Field-dependent dielectric converged in %d iterations (max diff = %e)\n",
        //     eps_iter, max_diff
        // );
    // }

    if (grid_ctx != NULL) {
        grid_ctx->eps_phi_iters = eps_iter;
    }

    if (res == -1) {
        fprintf(stderr, "Warning: Multigrid did not converge after %d iterations.\n", MG_ITER_LIMIT_PB);
    }

    return res;
}
