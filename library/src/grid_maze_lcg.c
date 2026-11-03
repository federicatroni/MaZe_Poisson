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
#include "multigrid.h"

void electrode_add_q(grid *g);   /* [electrode patch] grid_general.c */
void electrode_record(grid *g);

#ifdef __MPI
void maze_lcg_grid_init_mpi(grid *grid) {
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

void maze_lcg_grid_init_mpi(grid *grid) {
    mpi_data *mpid = get_mpi_data();
    mpid->n_loc = grid->n;
    mpid->n_start = 0;
}  // Do nothing

#endif  // __MPI

void maze_lcg_grid_init(grid * grid) {
    int n_loc = grid->n_local;
    int n = grid->n;

    long int n2 = n * n;

    maze_lcg_grid_init_mpi(grid);

    long int size = grid->n_local * n2;
    grid->size = size;

    grid->q = (double *)malloc(size * sizeof(double));
    grid->y = mpi_grid_allocate(n_loc, n);
    grid->phi_p = mpi_grid_allocate(n_loc, n);
    grid->phi_n = mpi_grid_allocate(n_loc, n);

    grid->init_field = maze_lcg_grid_init_field;
    grid->update_field = maze_lcg_grid_update_field;
    grid->update_charges = maze_lcg_grid_update_charges;

    switch (grid->precond_type) {
        case PRECOND_TYPE_BLOCKJACOBI:
            precond_blockjacobi_init();
            break;
        default:
            break;
    }
}

void maze_lcg_grid_cleanup(grid * grid) {
    free(grid->q);

    switch (grid->precond_type) {
        case PRECOND_TYPE_BLOCKJACOBI:
            precond_blockjacobi_cleanup();
            break;
        default:
            break;
    }

    mpi_grid_free(grid->y, grid->n);
    mpi_grid_free(grid->phi_p, grid->n);
    mpi_grid_free(grid->phi_n, grid->n);
}

void maze_lcg_grid_init_field(grid *grid) {
    long int i;

    double constant = -4 * M_PI / grid->h;

    if ( ! grid->pb_enabled) {
        constant /= grid->eps_s;  // Scale by the dielectric constant if not using PB explicitly
    }


    memset(grid->y, 0, grid->size * sizeof(double));  // y = 0
    memcpy(grid->phi_p, grid->phi_n, grid->size * sizeof(double));  // phi_prev = phi_n
    // phi_n = constant * q
    memcpy(grid->phi_n, grid->q, grid->size * sizeof(double));
    dscal(grid->phi_n, constant, grid->size);

    if (grid->pb_enabled) {
        conj_grad_pb(
            grid->phi_n, grid->y, grid->phi_n, grid->tol, grid->n_local, grid->n,
            grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
        );
    } else {
        conj_grad(grid->phi_n, grid->y, grid->phi_n, grid->tol, grid->n_local, grid->n);
    }
    electrode_record(grid);   /* [electrode patch] */
}

int maze_lcg_grid_update_field(grid *grid) {
    void (*precond)(double *, double *, int, int, int);

    switch (grid->precond_type) {
        case PRECOND_TYPE_NONE:
            precond = NULL;
            break;
        case PRECOND_TYPE_JACOBI:
            precond = precond_jacobi_apply;
            break;
        case PRECOND_TYPE_MG:
            precond = precond_mg_apply;
            break;
        case PRECOND_TYPE_SSOR:
            precond = precond_ssor_apply;
            break;
        case PRECOND_TYPE_BLOCKJACOBI:
            precond = precond_blockjacobi_apply;
            break;
        default:
            break;
    }

    int res;

    if (grid->pb_enabled) {
        res = verlet_poisson_pb(
            grid->tol, grid->h, grid->phi_n, grid->phi_p, grid->q, grid->y,
            grid->n_local, grid->n,
            grid->eps_x, grid->eps_y, grid->eps_z, grid->k2
        );
    } else {
        // Using fact that h is only used in (4 * pi / h) factor so multiplying by eps means:
        // sigma_p = M.phi + 4 * pi * rho / h  ->  sigma_p = M.phi + 4 * pi * rho / (h * eps)
        res = verlet_poisson(
            grid->tol, grid->h * grid->eps_s, grid->phi_n, grid->phi_p, grid->q, grid->y,
            grid->n_local, grid->n,
            precond
        );
    }
    electrode_record(grid);   /* [electrode patch] */
    return res;
}   

double maze_lcg_grid_update_charges(grid *grid, particles *p) {
    double r = update_charges(
        grid->n, p->n_p, grid->h, p->num_neighbors,
        p->pos, p->neighbors, p->charges, grid->q,
        p->charges_spread_func
    );
    electrode_add_q(grid);   /* [electrode patch] */
    return r;
}
