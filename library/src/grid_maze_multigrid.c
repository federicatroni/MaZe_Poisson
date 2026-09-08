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

static void init_y_history_buffers(grid *grid, int n_loc, int n, long int size) {
    for (int yh = 0; yh <= MAZE_Y_HIST_MAX; yh++) {
        grid->y_hist[yh] = mpi_grid_allocate(n_loc, n);
        memset(grid->y_hist[yh], 0, size * sizeof(double));
    }
}

static void reset_y_history_buffers(grid *grid) {
    for (int yh = 0; yh <= MAZE_Y_HIST_MAX; yh++) {
        memset(grid->y_hist[yh], 0, grid->size * sizeof(double));
    }
    grid->y_hist_len = 0;
}

static void free_y_history_buffers(grid *grid) {
    for (int yh = 0; yh <= MAZE_Y_HIST_MAX; yh++) {
        mpi_grid_free(grid->y_hist[yh], grid->n);
    }
}

void maze_multigrid_grid_init(grid * grid) {
    int n = grid->n;
    long int n2 = n * n;

    /* The Poisson solve remains entirely real-space.  The FFT-compatible
     * slab decomposition is retained only for optional charge smoothing. */
    grid_init_mpi_fft(grid);
    int n_loc = grid->n_local;

    long int size = grid->n_local * n2;
    grid->size = size;

    grid->q = mpi_grid_allocate(n_loc, n);
    grid->y = mpi_grid_allocate(n_loc, n);
    init_y_history_buffers(grid, n_loc, n, size);
    grid->phi_p = mpi_grid_allocate(n_loc, n);
    grid->phi_n = mpi_grid_allocate(n_loc, n);

    memset(grid->phi_p, 0, size * sizeof(double));  // phi_p = 0
    memset(grid->phi_n, 0, size * sizeof(double));  // phi_n = 0

    grid->init_field = maze_multigrid_grid_init_field;
    grid->update_field = maze_multigrid_grid_update_field;
    grid->update_charges = maze_multigrid_grid_update_charges;
}

void maze_multigrid_grid_cleanup(grid * grid) {
    mpi_grid_free(grid->q, grid->n);
    mpi_grid_free(grid->y, grid->n);
    free_y_history_buffers(grid);
    mpi_grid_free(grid->phi_p, grid->n);
    mpi_grid_free(grid->phi_n, grid->n);
}

void maze_multigrid_grid_init_field(grid *grid) {
    double *tmp = mpi_grid_allocate(grid->n_local, grid->n);

    double constant = -4 * M_PI / grid->h;
    if ( ! grid->pb_enabled) {
        constant /= grid->eps_s;  // Scale by the dielectric constant if not using PB explicitly
    }

    memset(grid->y, 0, grid->size * sizeof(double));  // y = 0
    reset_y_history_buffers(grid);
    vec_copy(grid->phi_n, grid->phi_p, grid->size);  // phi_prev = phi_n
    // phi_n = consant * q
    laplace_filter_rhs(grid->q, tmp, grid->n_local, grid->n);
    dscal(tmp, constant, grid->size);

    if (grid->pb_enabled) {
        multigrid_solve_pb(
            grid->tol, tmp, grid->phi_n, grid->n_local, grid->n, grid->n_start,
            grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
        );
    } else {
        multigrid_solve(
            grid->tol, tmp, grid->phi_n, grid->n_local, grid->n, grid->n_start
        );
    }

    mpi_grid_free(tmp, grid->n);
}

int maze_multigrid_grid_update_field(grid *grid) {
    int res;

    if (grid->pb_enabled) {
        if (grid->eps_field_dep_enabled) {
            res = verlet_pb_multigrid_eps_field(
                grid->tol, grid->h, grid->phi_n, grid->phi_p, grid->q, grid->y,
                grid->n_local, grid->n, grid->eps_x, grid->eps_y, grid->eps_z, grid->k2,
                grid
            );
        } else {
            res = verlet_pb_multigrid(
                grid->tol, grid->h, grid->phi_n, grid->phi_p, grid->q, grid->y,
                grid->y_hist, grid->y_extrap_order, &grid->y_hist_len,
                grid->n_local, grid->n, grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
            );
        }
    } else{
        res = verlet_poisson_multigrid(
            grid->tol, grid->h * grid->eps_s, grid->phi_n, grid->phi_p, grid->q, grid->y,
            grid->y_hist, grid->y_extrap_order, &grid->y_hist_len,
            grid->n_local, grid->n
        );  // grid->h * grid->eps_s to account for the dielectric constant in the poisson equation
    }
    if (grid->precond_type != PRECOND_TYPE_NONE) {
        fprintf(stderr, "Maze Multigrid with preconditioner not implemented yet.\n");
        exit(1);
    }

    return res;
}   

double maze_multigrid_grid_update_charges(grid *grid, particles *p) {
    return update_charges(
        grid->n, p->n_p, grid->h, p->num_neighbors,
        p->pos, p->grid_neighbors, p->charges, grid->q,
        p->charges_spread_func
    );
}
