#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "constants.h"
#include "charges.h"
#include "verlet.h"
#include "mp_structs.h"
#include "mpi_base.h"

static const double PHI_POLY_COEF[MAZE_PHI_HIST_MAX + 1][MAZE_PHI_HIST_MAX + 1] = {
    {1.0},
    {2.0, -1.0},
    {3.0, -3.0, 1.0},
    {4.0, -6.0, 4.0, -1.0},
    {5.0, -10.0, 10.0, -5.0, 1.0},
};

static void phi_push_current(grid *grid) {
    double *new_phi_p = grid->phi_hist[MAZE_PHI_HIST_MAX - 2];
    for (int i = MAZE_PHI_HIST_MAX - 2; i > 0; i--) {
        grid->phi_hist[i] = grid->phi_hist[i - 1];
    }
    grid->phi_hist[0] = grid->phi_p;
    grid->phi_p = new_phi_p;
    vec_copy(grid->phi_n, grid->phi_p, grid->size);
    if (grid->phi_hist_len < MAZE_PHI_HIST_MAX) grid->phi_hist_len++;
}

static void phi_build_guess(grid *grid) {
    double *current = grid->phi_hist[MAZE_PHI_HIST_MAX - 1];
    vec_copy(grid->phi_n, current, grid->size);

    int order = grid->phi_extrap_order;
    if (order > grid->phi_hist_len) order = grid->phi_hist_len;
    vec_copy(current, grid->phi_n, grid->size);
    dscal(grid->phi_n, PHI_POLY_COEF[order][0], grid->size);
    if (order >= 1) daxpy(grid->phi_p, grid->phi_n, PHI_POLY_COEF[order][1], grid->size);
    for (int j = 2; j <= order; j++) {
        daxpy(grid->phi_hist[j - 2], grid->phi_n, PHI_POLY_COEF[order][j], grid->size);
    }

    double *oldest = grid->phi_hist[MAZE_PHI_HIST_MAX - 2];
    for (int i = MAZE_PHI_HIST_MAX - 2; i > 0; i--) {
        grid->phi_hist[i] = grid->phi_hist[i - 1];
    }
    grid->phi_hist[0] = grid->phi_p;
    grid->phi_p = current;
    grid->phi_hist[MAZE_PHI_HIST_MAX - 1] = oldest;
    if (grid->phi_hist_len < MAZE_PHI_HIST_MAX) grid->phi_hist_len++;
}

void multigrid_grid_init(grid * grid) {
    int n_loc = grid->n_local;
    int n = grid->n;
    long int n2 = n * n;

    // TODO: revert this done for testing
    // grid_init_mpi(grid);
    grid_init_mpi_fft(grid);

    long int size = grid->n_local * n2;
    grid->size = size;

    grid->q = mpi_grid_allocate(n_loc, n);
    grid->y = mpi_grid_allocate(n_loc, n);
    grid->phi_p = mpi_grid_allocate(n_loc, n);
    grid->phi_n = mpi_grid_allocate(n_loc, n);
    for (int ph = 0; ph < MAZE_PHI_HIST_MAX; ph++) {
        grid->phi_hist[ph] = mpi_grid_allocate(n_loc, n);
        memset(grid->phi_hist[ph], 0, size * sizeof(double));
    }

    memset(grid->phi_p, 0, size * sizeof(double));  // phi_p = 0
    memset(grid->phi_n, 0, size * sizeof(double));  // phi_n = 0

    grid->init_field = multigrid_grid_init_field;
    grid->update_field = multigrid_grid_update_field;
    grid->update_charges = multigrid_grid_update_charges;
}

void multigrid_grid_cleanup(grid * grid) {
    mpi_grid_free(grid->q, grid->n);
    mpi_grid_free(grid->y, grid->n);
    mpi_grid_free(grid->phi_p, grid->n);
    mpi_grid_free(grid->phi_n, grid->n);
    for (int ph = 0; ph < MAZE_PHI_HIST_MAX; ph++) {
        mpi_grid_free(grid->phi_hist[ph], grid->n);
    }
}

void multigrid_grid_init_field(grid *grid) {
    double *rhs = mpi_grid_allocate(grid->n_local, grid->n);

    double constant = -4 * M_PI / grid->h;
    if ( ! grid->pb_enabled) {
        constant /= grid->eps_s;  // Scale by the dielectric constant if not using PB explicitly
    }

    memset(grid->y, 0, grid->size * sizeof(double));  // y = 0
    if (grid->phi_initialized) phi_push_current(grid);
    // Build the right-hand side separately so that phi_n remains the initial
    // guess and is overwritten in place by the converged potential.
    memcpy(rhs, grid->q, grid->size * sizeof(double));
    dscal(rhs, constant, grid->size);

    if (grid->pb_enabled) {
        multigrid_solve_pb(
            grid->tol, rhs, grid->phi_n, grid->n_local, grid->n, grid->n_start,
            grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
        );
    } else {
        multigrid_solve(
            grid->tol, rhs, grid->phi_n, grid->n_local, grid->n, grid->n_start
        );
    }

    grid->phi_initialized = 1;

    mpi_grid_free(rhs, grid->n);
}

int multigrid_grid_update_field(grid *grid) {
    int res = -1;
    double tol = grid->tol;

    double *tmp = mpi_grid_allocate(grid->n_local, grid->n);

    phi_build_guess(grid);
    // memset(grid->phi_n, 0, grid->size * sizeof(double));  // phi_n = 0 in case we need want multigrid to start without initial guess

    double constant = -4 * M_PI / grid->h;
    if ( ! grid->pb_enabled) {
        constant /= grid->eps_s;  // Scale by the dielectric constant if not using PB explicitly
    }
    
    // phi_n = constant * q
    vec_copy(grid->q, tmp, grid->size);
    dscal(tmp, constant, grid->size);

    // if poisson boltzmann is enabled use the pb multigrid solver, otherwise use the poisson one.
    // the RHS of the equation is always the same, what changes are the multigrid and the laplace_filter functions
    if (grid->pb_enabled) {
        res = multigrid_solve_pb(
            tol, tmp, grid->phi_n, grid->n_local, grid->n, grid->n_start,
            grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
        );
    } 
    else{
        res = multigrid_solve(tol, tmp, grid->phi_n, grid->n_local, grid->n, grid->n_start);
    }

    mpi_grid_free(tmp, grid->n);

    return res;
}   

double multigrid_grid_update_charges(grid *grid, particles *p) {
    return update_charges(
        grid->n, p->n_p, grid->h, p->num_neighbors,
        p->pos, p->grid_neighbors, p->charges, grid->q,
        p->charges_spread_func
    );
}
