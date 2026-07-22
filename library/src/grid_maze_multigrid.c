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

#define PB_GAUGE_FIX_STRIDE 1000L

static int pb_operator_has_constant_zero_mode(const grid *g) {
    /* The PB stencil used here is periodic.  At zero ionic strength k2 is
       identically zero, so adding a constant to the potential changes no
       physical field and belongs to the null space of the operator. */
    return g->pb_enabled && g->kbar2 == 0.0;
}

static void center_pb_verlet_fields(grid *g) {
    const long int global_size = (long int)g->n * g->n * g->n;
    remove_global_mean(g->phi_n, g->size, global_size);
    remove_global_mean(g->phi_p, g->size, global_size);
}

#ifdef __MPI
void maze_multigrid_grid_init_mpi(grid *grid) {
    mpi_data *mpid = get_mpi_data();

    int n = grid->n;
    int rank = mpid->rank;
    int size = mpid->size;

    int div, mod;
    int n_loc, n_start;

    div = n / size;
    mod = n % size;
    for (int i=0; i<size; i++) {
        if (i < mod) {
            n_loc = div + 1;
            n_start = i * n_loc;
        } else {
            n_loc = div;
            n_start = i * n_loc + mod;
        }
        mpid->n_loc_list[i] = n_loc;
        mpid->n_start_list[i] = n_start;
    }

    grid->n_local = mpid->n_loc_list[rank];
    grid->n_start = mpid->n_start_list[rank];
    mpid->n_loc = grid->n_local;
    mpid->n_start = grid->n_start;
}

#else  // __MPI

void maze_multigrid_grid_init_mpi(grid *grid) {
    mpi_data *mpid = get_mpi_data();
    mpid->n_loc = grid->n;
    mpid->n_start = 0;
}  // Do nothing

#endif  // __MPI

void maze_multigrid_grid_init(grid * grid) {
    int n_loc = grid->n_local;
    int n = grid->n;

    long int n2 = n * n;

    maze_multigrid_grid_init_mpi(grid);

    long int size = grid->n_local * n2;
    grid->size = size;

    grid->q = (double *)malloc(size * sizeof(double));
    grid->y = mpi_grid_allocate(n_loc, n);
    grid->phi_p = mpi_grid_allocate(n_loc, n);
    grid->phi_n = mpi_grid_allocate(n_loc, n);

    memset(grid->phi_p, 0, size * sizeof(double));  // phi_p = 0
    memset(grid->phi_n, 0, size * sizeof(double));  // phi_n = 0

    grid->init_field = maze_multigrid_grid_init_field;
    grid->update_field = maze_multigrid_grid_update_field;
    grid->update_charges = maze_multigrid_grid_update_charges;
}

void maze_multigrid_grid_cleanup(grid * grid) {
    free(grid->q);

    mpi_grid_free(grid->y, grid->n);
    mpi_grid_free(grid->phi_p, grid->n);
    mpi_grid_free(grid->phi_n, grid->n);
}

void maze_multigrid_grid_init_field(grid *grid) {
    long int i;

    double *tmp = mpi_grid_allocate(grid->n_local, grid->n);

    double constant = -4 * M_PI / grid->h;
    if ( ! grid->pb_enabled) {
        constant /= grid->eps_s;  // Scale by the dielectric constant if not using PB explicitly
    }

    memset(grid->y, 0, grid->size * sizeof(double));  // y = 0
    vec_copy(grid->phi_n, grid->phi_p, grid->size);  // phi_prev = phi_n
    // phi_n = consant * q
    vec_copy(grid->q, tmp, grid->size);
    dscal(tmp, constant, grid->size);

    if (grid->pb_enabled) {
        conj_grad_pb(
            tmp, grid->y, grid->phi_n, grid->tol, grid->n_local, grid->n,
            grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
        );
    } else {
        conj_grad(tmp, grid->y, grid->phi_n, grid->tol, grid->n_local, grid->n);
    }

    if (pb_operator_has_constant_zero_mode(grid)) {
        center_pb_verlet_fields(grid);
    }

    mpi_grid_free(tmp, grid->n);
}

int maze_multigrid_grid_update_field(grid *grid) {
    int precond = 1;

    switch (grid->precond_type) {
        case PRECOND_TYPE_NONE:
            precond = 0;
            break;
        default:
            break;
    }

    int res;
    const int zero_mode = pb_operator_has_constant_zero_mode(grid);

    if (zero_mode) {
        grid->field_update_count++;
        if (grid->field_update_count % PB_GAUGE_FIX_STRIDE == 0) {
            center_pb_verlet_fields(grid);
        }
    }

    if (grid->pb_enabled) {
        if (grid->eps_field_dep_enabled) {
            res = verlet_pb_multigrid_eps_field(
                grid->tol, grid->h, grid->phi_n, grid->phi_p, grid->q, grid->y,
                grid->n_local, grid->n, grid->eps_x, grid->eps_y, grid->eps_z, grid->k2,
                grid
            );
        } else {
            res = verlet_pb_multigrid_with_retry(
                grid->tol, grid->h, grid->phi_n, grid->phi_p, grid->q, grid->y,
                grid->n_local, grid->n, grid->eps_x, grid->eps_y, grid->eps_z, grid->k2,
                zero_mode
            );
        }
    } else{
        res = verlet_poisson_multigrid(
            grid->tol, grid->h * grid->eps_s, grid->phi_n, grid->phi_p, grid->q, grid->y,
            grid->n_local, grid->n
        );  // grid->h * grid->eps_s to account for the dielectric constant in the poisson equation
    }
    if (precond) {
        fprintf(stderr, "Maze Multigrid with preconditioner not implemented yet.\n");
        exit(1);
    }

    return res;
}   

double maze_multigrid_grid_update_charges(grid *grid, particles *p) {
    return update_charges(
        grid->n, p->n_p, grid->h, p->num_neighbors,
        p->pos, p->neighbors, p->charges, grid->q,
        p->charges_spread_func
    );
}
