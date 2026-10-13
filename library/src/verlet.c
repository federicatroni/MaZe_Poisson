#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "verlet.h"
#include "constants.h"
#include "mpi_base.h"
#include "linalg.h"
#include "mp_structs.h"
#include "multigrid_semi.h"

#define EPS_FIELD_TOL 1e-1
#define EPS_FIELD_MAX_ITER 500

#ifdef __cplusplus
#define EXTERN_C extern "C"                                                           
#else
#define EXTERN_C
#endif

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

    long int i;
    long int n2 = size2 * size2;
    long int n3 = size1 * n2;

    double app;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    
    // Compute provisional update for the field phi
    #pragma omp parallel for private(app)
    for (i = 0; i < n3; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }

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
    int size1, int size2
) {
    int res = -1;
    int iter_conv = 0;

    long int i;
    long int n3 = size1 * size2 * size2;

    double app, constant;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    double *tmp2 = (double*)malloc(n3 * sizeof(double));

    // Compute provisional update for the field phi
    #pragma omp parallel for private(app)
    for (i = 0; i < n3; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }

    constant = (4 * M_PI) / h;
    laplace_filter(phi, tmp2, size1, size2);
    daxpy(q, tmp2, constant, n3);  // sigma_p = A . phi + 4 * pi * rho / eps
    if (!mg_keep_y) memset(y, 0, n3 * sizeof(double));  // start from y = 0 (MAZE_MG_KEEP_Y=1: previous y)
    
    // Questo pezzo non e' usato se vedi app e tmp2 vengono riscritti prima di essere letti
    // Serve solo se abiliti il printf sotto
    // laplace_filter(y, tmp, size1, size2);
    // daxpy(tmp2, tmp, -1., n3);  // res = A . y - sigma_p
    // app = norm_inf(tmp, n3);   // Compute norm_inf of residual
    // printf("\ny = %e \t iter=%d \t res=%e\n", norm_inf(y, n3), iter_conv,app);
    
    while(iter_conv < MG_ITER_LIMIT) { 
        // Compute the constraint with the provisional value of the field phi
        multigrid_apply(tmp2, y, size1, size2, get_n_start(), MG_SOLVE_SM); //solve A . y = sigma_p

        laplace_filter(y, tmp, size1, size2);
        daxpy(tmp2, tmp, -1., n3);  // res = A . y - sigma_p
        app = norm_inf(tmp, n3);   // Compute norm_inf of residual
        iter_conv++;
        
        // printf("\ny = %e \t iter=%d \t res=%e\n", norm_inf(y, n3), iter_conv,app);
        
        if (app <= tol){
            res = iter_conv;
            break;
        }
    }
    // Scale the field with the constrained 'force' term
    daxpy(y, phi, -1.0, n3);  // phi = phi - y

    // Free temporary arrays
    free(tmp);
    free(tmp2);

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

    long int i;
    long int n2 = size2 * size2;
    long int n3 = size1 * n2;

    double app;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    
    // Compute provisional update for the field phi
    #pragma omp parallel for private(app)
    for (i = 0; i < n3; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }

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


/* Solve A_pb y = rhs with repeated multigrid cycles. */
static int solve_pb_multigrid_correction(
    double tol, double *rhs, double *y, double *residual,
    int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
    const long int n3 = (long int)size1 * size2 * size2;

    /* [slit_dsl patch] MAZE_MG_SEMI=1: replace the standard V-cycle by the semi-coarsening V-cycle
     * (multigrid_semi.c). Same fine operator, rhs and stopping test; single MPI rank only. */
    static int semi = -1;
    if (semi < 0) { const char *e = getenv("MAZE_MG_SEMI"); semi = (e != NULL && atoi(e) != 0 && get_size() == 1); }
    if (semi && semi_mg_setup(size1, size2, eps_x, eps_y, eps_z, k2_screen) == 0) semi = 0;  /* grid not coarsenable: standard V-cycle from now on */
#define MG_CYCLE(F, U) do { if (semi) semi_mg_vcycle((F), (U), size1, size2); \
        else multigrid_pb_apply((F), (U), size1, size2, get_n_start(), MG_SOLVE_SM_PB, eps_x, eps_y, eps_z, k2_screen); } while (0)
    int ret_all = -1;

    /* [slit_dsl patch] MAZE_MG_KRYLOV=1: flexible CG (FCG(1), Notay 2000) preconditioned by one
     * V-cycle (zero initial guess). Same operator, same rhs, same stopping test (max|A y - rhs| <= tol)
     * as the stationary V-cycle iteration below: only the number of V-cycles changes.
     * Returns the number of V-cycles, or -1 if not converged. */
    static int krylov = -1;
    if (krylov < 0) { const char *e = getenv("MAZE_MG_KRYLOV"); krylov = (e != NULL && atoi(e) != 0); }
    if (krylov) {
        double *r = mpi_grid_allocate(size1, size2);
        double *z = mpi_grid_allocate(size1, size2);
        double *p = mpi_grid_allocate(size1, size2);
        double *q = mpi_grid_allocate(size1, size2);
        double res_norm;
        int ret = -1;
        /* r = rhs - A y */
        laplace_filter_pb_residual(y, rhs, r, &res_norm, size1, size2, eps_x, eps_y, eps_z, k2_screen);
        if (res_norm <= tol) ret = 0;
        else {
            memset(z, 0, n3 * sizeof(double));
            MG_CYCLE(r, z);
            vec_copy(z, p, n3);
            for (int iteration = 1; iteration <= MG_ITER_LIMIT_PB; iteration++) {
                laplace_filter_pb(p, q, size1, size2, eps_x, eps_y, eps_z, k2_screen);   /* q = A p */
                const double pq = ddot(p, q, n3);
                const double alpha = ddot(z, r, n3) / pq;
                daxpy(p, y, alpha, n3);    /* y += alpha p */
                daxpy(q, r, -alpha, n3);   /* r -= alpha q */
                res_norm = norm_inf(r, n3);
                if (res_norm <= tol) { ret = iteration; break; }
                memset(z, 0, n3 * sizeof(double));
                MG_CYCLE(r, z);
                const double beta = -ddot(z, q, n3) / pq;   /* FCG(1): A-orthogonalise against previous p */
                dscal(p, beta, n3);
                daxpy(z, p, 1.0, n3);      /* p = z + beta p */
            }
        }
        mpi_grid_free(r, size2); mpi_grid_free(z, size2); mpi_grid_free(p, size2); mpi_grid_free(q, size2);
        if (semi) semi_mg_free(size2);
        return ret;
    }

    for (int iteration = 1; iteration <= MG_ITER_LIMIT_PB; iteration++) {
        MG_CYCLE(rhs, y);

        double res_norm;
        laplace_filter_pb_residual(
            y, rhs, NULL, &res_norm, size1, size2,
            eps_x, eps_y, eps_z, k2_screen
        );

        if (res_norm <= tol) {
            ret_all = iteration;
            break;
        }
    }

    if (semi) semi_mg_free(size2);
    return ret_all;
#undef MG_CYCLE
}


/*
Apply the MaZe Verlet update and PB constraint.  When zero_mode is true, a
failed solve is retried after removing the constant gauge component from the
two Verlet fields and resetting only the retry's initial y guess.
*/
EXTERN_C int verlet_pb_multigrid_with_retry(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2, double *eps_x, double *eps_y, double *eps_z, double *k2_screen,
    int zero_mode
) {
    long int i;
    long int n3 = (long int)size1 * size2 * size2;

    double app;
    double *tmp = (double*)malloc(n3 * sizeof(double));
    double *tmp2 = (double*)malloc(n3 * sizeof(double));

    // Compute provisional update for the field phi
    #pragma omp parallel for private(app)
    for (i = 0; i < n3; i++) {
        app = phi[i];
        phi[i] = 2 * app - phi_prev[i];
        phi_prev[i] = app;
    }

    const double constant = (4 * M_PI) / h;
    laplace_filter_pb(phi, tmp2, size1, size2, eps_x, eps_y, eps_z, k2_screen);
    daxpy(q, tmp2, constant, n3);  // sigma_p = A_pb . phi + 4 * pi * q / h

    /* Start from y = 0: the previous y is a worse guess (its residual is larger than that of zero) */
    memset(y, 0, n3 * sizeof(double));

    int res = solve_pb_multigrid_correction(
        tol, tmp2, y, tmp, size1, size2,
        eps_x, eps_y, eps_z, k2_screen
    );

    if (res == -1 && zero_mode) {
        /*
         * The periodic I=0 operator cannot determine a constant potential.
         * Remove only that null component from the two Verlet fields, then
         * retry the same constraint from a clean y guess.  A converged y is
         * never altered by this recovery path.
         */
        const long int global_size = (long int)size2 * size2 * size2;
        remove_global_mean(phi, n3, global_size);
        remove_global_mean(phi_prev, n3, global_size);
        memset(y, 0, n3 * sizeof(double));

        laplace_filter_pb(phi, tmp2, size1, size2, eps_x, eps_y, eps_z, k2_screen);
        daxpy(q, tmp2, constant, n3);

        res = solve_pb_multigrid_correction(
            tol, tmp2, y, tmp, size1, size2,
            eps_x, eps_y, eps_z, k2_screen
        );

        if (res != -1) {
            mpi_fprintf(
                stderr,
                "Warning: PB multigrid first attempt did not converge after %d iterations; "
                "gauge-centered retry converged in %d iterations.\n",
                MG_ITER_LIMIT_PB, res
            );
        }
    }

    if (res == -1) {
        /* Never apply an unconverged correction to the field or reuse it. */
        memset(y, 0, n3 * sizeof(double));
    } else {
        daxpy(y, phi, -1.0, n3);  // phi = phi - y
    }

    // Free temporary arrays
    free(tmp);
    free(tmp2);

    if (res == -1) {
        mpi_fprintf(
            stderr,
            "Warning: PB multigrid did not converge after %d iterations%s; "
            "discarded the unconverged correction and reset y to zero.\n",
            MG_ITER_LIMIT_PB,
            zero_mode ? " and a gauge-centered retry" : ""
        );
    }

    return res;
}

EXTERN_C int verlet_pb_multigrid(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2, double *eps_x, double *eps_y, double *eps_z, double *k2_screen
) {
    return verlet_pb_multigrid_with_retry(
        tol, h, phi, phi_prev, q, y,
        size1, size2, eps_x, eps_y, eps_z, k2_screen, 0
    );
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

            laplace_filter_pb_residual(y, tmp2, NULL, &app, size1, size2, eps_x, eps_y, eps_z, k2_screen);  // app = norm_inf(A . y - sigma_p)
            iter_conv++;
            
            // printf("\ny = %e \t iter=%d \t res=%e\n", norm_inf(y, n3), iter_conv,app);
            
            if (app <= tol){
                res = iter_conv;
                break;
            }
        }
        if (res == -1) {
            /* Do not apply or reuse a correction that failed to converge. */
            memset(y, 0, n3 * sizeof(double));
        } else {
            daxpy(y, phi, -1.0, n3);  // phi = phi - y
        }

        eps_iter++;
        // mpi_printf("Field-dependent dielectric iteration %d: max diff = %e\n", eps_iter, max_diff);

        /* Stop early if we were not actually iterating eps */
        if (res == -1 || grid_ctx == NULL || grid_ctx->kBT <= 0.0) {
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
        mpi_fprintf(
            stderr,
            "Warning: PB multigrid did not converge after %d iterations; "
            "discarded the unconverged correction and reset y to zero.\n",
            MG_ITER_LIMIT_PB
        );
    }

    return res;
}
