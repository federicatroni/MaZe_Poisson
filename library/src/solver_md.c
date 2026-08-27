#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "mp_structs.h"
#include "mpi_base.h"
#include "omp_base.h"

#define MAX_ITG_PARAMS 10

particles *g_particles = NULL;
integrator *g_integrator = NULL;
grid *g_grid = NULL;

double q_tot = 0.0;


void solver_initialize() {
    int size = init_mpi();
    int rank = get_rank();

    int n_threads = get_omp_max_threads();

    mpi_printf("******************************************************\n");
    mpi_printf("* MAZE_POISSON\n");
    if (size > 0) {
        mpi_printf("*   MPI     enabled  running on %d processes\n", size);
    } else {
        mpi_printf("*   MPI     not available\n");
    }
    if (n_threads > 0) {
        mpi_printf("*   OpenMP  enabled  running on %d threads\n", n_threads);
    } else {
        mpi_printf("*   OpenMP  not available\n");
    }
    mpi_printf("******************************************************\n");
}

void solver_initialize_grid(
    int n_grid, double L, double h, double tol, double eps, double eps_int, int grid_type, int precond_type
) {
    g_grid = grid_init(n_grid, L, h, tol, eps, eps_int, grid_type, precond_type);
}

void solver_initialize_grid_pois_boltz(
    double w, double kbar2, int nonpolar_enabled, int eps_map_type, int pb_force_type, int stress_tensor_bc_type, double kBT, double eps_field_alpha
) {
    // Initialize the solvent potential and dielectric constant arrays
    grid_pb_init(g_grid, w, kbar2, nonpolar_enabled, eps_map_type, pb_force_type, stress_tensor_bc_type, kBT, eps_field_alpha);
}

void solver_initialize_particles(
    int n, int n_typ, double L, double h, int n_p, int pot_type, int cas_type,
    int *types, double *pos, double *vel, double *mass, double *charges,
    double *pot_params
) {
    g_particles = particles_init(n, n_p, n_typ, L, h, cas_type);

    memcpy(g_particles->types, types, n_p * sizeof(int));
    memcpy(g_particles->pos, pos, n_p * 3 * sizeof(double));
    memcpy(g_particles->vel, vel, n_p * 3 * sizeof(double));
    memcpy(g_particles->mass, mass, n_p * sizeof(double));
    memcpy(g_particles->charges, charges, n_p * sizeof(double));
    
    g_particles->init_potential(g_particles, pot_type, pot_params);
}

void solver_initialize_particles_pois_boltz(double gamma_np, double beta_np, double *solv_radii) {
    particles_pb_init(g_particles, gamma_np, beta_np, solv_radii);
}

void solver_initialize_sphere_pairwise_harmonic(
    double eps_s, double eps_int, double *radii
) {
    particles_init_sphere_pairwise_harmonic(g_particles, eps_s, eps_int, radii);
}

void solver_initialize_integrator(int n_p, double dt, double T, double gamma, int itg_type, int itg_enabled) {
    g_integrator = integrator_init(n_p, dt, itg_type);

    double itg_params[MAX_ITG_PARAMS];
    itg_params[0] = T;
    switch (itg_type) {
        case INTEGRATOR_TYPE_OVRVO:
            itg_params[1] = gamma;
            break;
        case INTEGRATOR_TYPE_VERLET:
            break;
        default:
            break;
    }

    if (itg_enabled == 1) {
        g_integrator->init_thermostat(g_integrator, itg_params);
    }
}

int solver_update_charges() {
    int res = 0;
    double q_tot_loc;
    
    g_particles->update_nearest_neighbors(g_particles);
    q_tot_loc = g_grid->update_charges(g_grid, g_particles);

    if (fabs(q_tot - q_tot_loc) > 1e-6) {
        res = 1;
        printf("Charge conservation error: q_tot = %.6f, q_tot_loc = %.6f\n", q_tot, q_tot_loc);
        exit(1);
    }

    q_tot = q_tot_loc;

    return res;
}

void solver_init_field() {
    g_grid->init_field(g_grid);
}

void solver_set_field(double *phi) {
    int n = g_grid->n;
    long int n2 = n * n;
    int n_start = get_n_start();

    memcpy(g_grid->phi_n, phi + n_start * n2, g_grid->size * sizeof(double));
}

void solver_set_field_prev(double *phi) {
    if (g_grid->phi_p != NULL) {
        int n = g_grid->n;
        long int n2 = n * n;
        int n_start = get_n_start();

        memcpy(g_grid->phi_p, phi + n_start * n2, g_grid->size * sizeof(double));
    }
}

int solver_update_field() {
    return g_grid->update_field(g_grid);
}

void solver_update_eps_k2() {
    // Update the dielectric constant and screening factor based on the grid's transition state
    g_grid->update_eps_and_k2(g_grid, g_particles);
}

int get_eps_phi_iters() {
    return g_grid ? g_grid->eps_phi_iters : 0;
}

void solver_compute_forces_elec() {
    g_particles->compute_forces_field(g_particles, g_grid);
}

double solver_compute_forces_sphere_pairwise_harmonic() {
    return particles_compute_forces_sphere_pairwise_harmonic(g_particles);
}

double solver_compute_forces_noel() {
    return g_particles->compute_forces_noel(g_particles);
}

double solver_compute_forces_pb() {
    return g_particles->compute_forces_pb(g_particles, g_grid);
}

void solver_compute_forces_tot() {
    g_particles->compute_forces_tot(g_particles);
}

// void solver_compute_forces() {
//     solver_compute_forces_elec();
//     solver_compute_forces_noel();
//     solver_compute_forces_tot();
// }

void integrator_part_1() {
    g_integrator->part1(g_integrator, g_particles);
}

void integrator_part_2() {
    g_integrator->part2(g_integrator, g_particles);
}

void solver_rescale_velocities() {
    g_particles->rescale_velocities(g_particles);
}

// int solver_initialize_md(int preconditioning, int vel_rescale) {
//     int res = 0;

//     #pragma omp parallel for reduction(+:q_tot)
//     for (int i = 0; i < g_particles->n_p; i++) {
//         q_tot += g_particles->charges[i];
//     }

//     // Step 0 Verlet
//     res |= solver_update_charges();
//     if (preconditioning == 1)
//         solver_init_field();
//     solver_compute_forces();

//     // Step 1 Verlet
//     integrator_part_1();
//     res |= solver_update_charges();
//     if (preconditioning == 1) 
//         solver_init_field();
//     solver_compute_forces();
//     integrator_part_2();

//     if (vel_rescale == 1) 
//         solver_rescale_velocities();

//     return res;
// }

// void solver_md_loop_iter() {
//     integrator_part_1();
//     solver_update_charges();
//     solver_update_field();
//     solver_compute_forces();
//     integrator_part_2();
// }

int solver_check_thermostat() {
    int res = 0;
    double temp;
    if (g_integrator->enabled == INTEGRATOR_ENABLED) {
        temp = g_particles->get_temperature(g_particles);
        if (fabs(temp - g_integrator->T) < 100) {
            res = 1;
            g_integrator->stop_thermostat(g_integrator);
        }
    }

    return res;
}

// void solver_run_n_steps(int n_steps) {
//     for (int i = 0; i < n_steps; i++) {
//         solver_md_loop_iter();
//     }
// }

void solver_finalize() {
    if (g_particles != NULL) {
        g_particles->free(g_particles);
        g_particles = NULL;
    }
    if (g_grid != NULL) {
        g_grid->free(g_grid);
        g_grid = NULL;
    }
    if (g_integrator != NULL) {
        g_integrator->free(g_integrator);
        g_integrator = NULL;
    }

    cleanup_mpi();
}

void get_pos(double *recv) {
    memcpy(recv, g_particles->pos, g_particles->n_p * 3 * sizeof(double));
}

void get_vel(double *recv) {
    memcpy(recv, g_particles->vel, g_particles->n_p * 3 * sizeof(double));
}

void get_fcs_elec(double *recv) {
    memcpy(recv, g_particles->fcs_elec, g_particles->n_p * 3 * sizeof(double));
}

void get_fcs_noel(double *recv) {
    memcpy(recv, g_particles->fcs_noel, g_particles->n_p * 3 * sizeof(double));
}

void get_fcs_db(double *recv) {
    if (g_particles->fcs_db != NULL) {
        memcpy(recv, g_particles->fcs_db, g_particles->n_p * 3 * sizeof(double));
    } else {
        memset(recv, 0, g_particles->n_p * 3 * sizeof(double));
    }
}

void get_fcs_ib(double *recv) {
    if (g_particles->fcs_ib != NULL) {
        memcpy(recv, g_particles->fcs_ib, g_particles->n_p * 3 * sizeof(double));
    } else {
        memset(recv, 0, g_particles->n_p * 3 * sizeof(double));
    }
}

void get_fcs_np(double *recv) {
    if (g_particles->fcs_np != NULL) {
        memcpy(recv, g_particles->fcs_np, g_particles->n_p * 3 * sizeof(double));
    } else {
        memset(recv, 0, g_particles->n_p * 3 * sizeof(double));
    }
}

void get_fcs_tot(double *recv) {
    memcpy(recv, g_particles->fcs_tot, g_particles->n_p * 3 * sizeof(double));
}

void get_types(int *recv) {
    memcpy(recv, g_particles->types, g_particles->n_p * sizeof(int));
}

void get_charges(double *recv) {
    memcpy(recv, g_particles->charges, g_particles->n_p * sizeof(double));
}

void get_masses(double *recv) {
    memcpy(recv, g_particles->mass, g_particles->n_p * sizeof(double));
}

void get_radii(double *recv) {
    if (g_particles->solv_radii != NULL) {
        memcpy(recv, g_particles->solv_radii, g_particles->n_p * sizeof(double));
    } else {
        memset(recv, 0, g_particles->n_p * sizeof(double));
    }
}

void get_field(double *recv) {
    mpi_grid_collect_buffer(g_grid->phi_n, recv, g_grid->n);
}

void get_field_prev(double *recv) {
    double *ptr = g_grid->phi_p !=  NULL ? g_grid->phi_p : g_grid->phi_n;
    mpi_grid_collect_buffer(ptr, recv, g_grid->n);
}

void get_eps_map(double *recv_x, double *recv_y, double *recv_z) {
    long int n3 = (long) g_grid->n * g_grid->n * g_grid->n;


    // TODO would leave this comment here to mark that this function is not really implemented yet,
    mpi_fprintf(stderr, "get_eps_map is not fully implemented yet, returning zeros\n");
    if (recv_x != NULL) {
        memset(recv_x, 0, n3 * sizeof(double));
    }

    if (recv_y != NULL) {
        memset(recv_y, 0, n3 * sizeof(double));
    }

    if (recv_z != NULL) {
        memset(recv_z, 0, n3 * sizeof(double));
    }
}

// void get_field_s(double *recv) {
//     if (g_grid->phi_s != NULL) {
//         mpi_grid_collect_buffer(g_grid->phi_s, recv, g_grid->n);
//     } else {
//         memset(recv, 0, g_grid->n * sizeof(double));
//     }
// }

// void get_phi_s_prev(double *recv) {
//     if (g_grid->phi_s_prev != NULL) {
//         mpi_grid_collect_buffer(g_grid->phi_s_prev, recv, g_grid->n);
//     } else {
//         memset(recv, 0, g_grid->n * sizeof(double));
//     }
// }

void get_q(double *recv) {
    mpi_grid_collect_buffer(g_grid->q, recv, g_grid->n);
}

double get_kinetic_energy() {
    return g_particles->get_kinetic_energy(g_particles);
}

double get_energy_elec() {
    return grid_get_energy_elec(g_grid);
}

void get_momentum(double *recv) {
    g_particles->get_momentum(g_particles, recv);
}

double get_temperature() {
    return g_particles->get_temperature(g_particles);
}
