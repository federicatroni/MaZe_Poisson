#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "charges.h"
#include "constants.h"
#include "forces.h"
#include "mp_structs.h"
#include "mpi_base.h"

#define NUM_NEIGH_CIC 8
#define NUM_NEIGH_SPLINE 64

// Potential types
char potential_type_str[PARTICLE_POTENTIAL_TYPE_NUM][16] = {"TF", "LJ", "SC"};

int get_potential_type_num() {
    return PARTICLE_POTENTIAL_TYPE_NUM;
}

char *get_potential_type_str(int n) {
    return potential_type_str[n];
}

// Charge assignment scheme types
char ca_scheme_type_str[CHARGE_ASS_SCHEME_TYPE_NUM][16] = {"CIC", "SPL_QUADR", "SPL_CUBIC"};

int get_ca_scheme_type_num() {
    return CHARGE_ASS_SCHEME_TYPE_NUM;
}

char *get_ca_scheme_type_str(int n) {
    return ca_scheme_type_str[n];
}

void particle_charges_init(particles *p, int cas_type) {
    int n_p = p->n_p;

    p->cas_type = cas_type;
    switch (cas_type) {
        case CHARGE_ASS_SCHEME_TYPE_CIC:
            p->num_neighbors = NUM_NEIGH_CIC;
            p->neighbors = (long int *)malloc(n_p * NUM_NEIGH_CIC * 3 * sizeof(long int));
            p->update_nearest_neighbors = particles_update_nearest_neighbors_cic;
            p->charges_spread_func = spread_cic;
            break;
        case CHARGE_ASS_SCHEME_TYPE_SPLQUAD:
            p->num_neighbors = NUM_NEIGH_SPLINE;
            p->neighbors = (long int *)malloc(n_p * NUM_NEIGH_SPLINE * 3 * sizeof(long int));
            p->update_nearest_neighbors = particles_update_nearest_neighbors_spline;
            p->charges_spread_func = spread_spline_quadr;
            break;
        case CHARGE_ASS_SCHEME_TYPE_SPLCUB:
            p->num_neighbors = NUM_NEIGH_SPLINE;
            p->neighbors = (long int *)malloc(n_p * NUM_NEIGH_SPLINE * 3 * sizeof(long int));
            p->update_nearest_neighbors = particles_update_nearest_neighbors_spline;
            p->charges_spread_func = spread_spline_cubic;
            break;
        default:
            mpi_fprintf(stderr, "Invalid charge assignment scheme type %d\n", cas_type);
            exit(1);
            break;
    }   
}

particles * particles_init(int n, int n_p, int n_typ, double L, double h, int cas_type) {
    particles *p = (particles *)malloc(sizeof(particles));
    p->n = n;
    p->n_p = n_p;
    p->n_typ = n_typ;
    p->L = L;
    p->h = h;

    p->types = (int *)malloc(n_p * sizeof(int));
    p->pos = (double *)malloc(n_p * 3 * sizeof(double));
    p->vel = (double *)malloc(n_p * 3 * sizeof(double));
    p->fcs_elec = (double *)calloc(n_p * 3, sizeof(double));
    p->fcs_noel = (double *)calloc(n_p * 3, sizeof(double));
    p->fcs_tot = (double *)calloc(n_p * 3, sizeof(double));
    p->mass = (double *)malloc(n_p * sizeof(double));
    p->charges = (double *)malloc(n_p * sizeof(double));

    p->pb_enabled = 0;  // Poisson-Boltzmann not enabled by default
    p->fcs_db = NULL;
    p->fcs_ib = NULL;
    p->fcs_np = NULL;
    p->solv_radii = NULL;

    // p->neighbors = (long int *)malloc(n_p * 24 * sizeof(long int));
    particle_charges_init(p, cas_type);

    p->tf_params = NULL;
    p->lj_params = NULL;
    p->sc_params = NULL;
    p->pairwise_dielectric_model = PAIRWISE_DIELECTRIC_MODEL_NONE;
    p->pairwise_dielectric_radii = NULL;

    p->free = particles_free;
    p->init_potential = particles_init_potential;
    
    p->compute_forces_field = particles_compute_forces_field;
    p->compute_forces_noel = NULL;
    p->compute_forces_tot = particles_compute_forces_tot;
    p->get_temperature = particles_get_temperature;
    p->get_kinetic_energy = particles_get_kinetic_energy;
    p->get_momentum = particles_get_momentum;
    p->rescale_velocities = particles_rescale_velocities;

    return p;
}

void particles_pb_init(particles *p, double gamma_np, double beta_np, double *solv_radii) {
    p->pb_enabled = 1;  // Enable Poisson-Boltzmann
    p->gamma_np = gamma_np;
    p->beta_np = beta_np;

    p->compute_forces_pb = particles_compute_forces_pb;

    p->solv_radii = (double *)malloc(p->n_p * sizeof(double));
    memcpy(p->solv_radii, solv_radii, p->n_p * sizeof(double));

    // Allocate forces for dielectric and ionic boundary conditions
    p->fcs_np = (double *)calloc(p->n_p * 3, sizeof(double));  // Non-polar forces
    p->fcs_db = (double *)calloc(p->n_p * 3, sizeof(double));  // Dielectric boundary forces
    p->fcs_ib = (double *)calloc(p->n_p * 3, sizeof(double));  // Ionic boundary forces
    
    // // Initialize the potential for Poisson-Boltzmann
    // p->init_potential(p, PARTICLE_POTENTIAL_TYPE_LD);
}

void particles_pb_free(particles *p) {
    if (p->pb_enabled) {
        free(p->solv_radii);
        free(p->fcs_db);
        free(p->fcs_ib);
        free(p->fcs_np);
    }
}

void particles_free(particles *p) {
    free(p->types);
    free(p->pos);
    free(p->vel);
    free(p->fcs_elec);
    free(p->fcs_noel);
    free(p->fcs_tot);
    free(p->mass);
    free(p->charges);
    free(p->neighbors);
    if (p->tf_params != NULL) {
        free(p->tf_params);
    }
    if (p->sc_params != NULL) {
        free(p->sc_params);
    }
    if (p->lj_params != NULL) {
        free(p->lj_params);
    }
    if (p->pairwise_dielectric_radii != NULL) {
        free(p->pairwise_dielectric_radii);
    }

    particles_pb_free(p);

    free(p);
}

void particles_init_potential(particles *p, int pot_type, double *pot_params) {
    p->pot_type = pot_type;
    switch (pot_type) 
    {
    case PARTICLE_POTENTIAL_TYPE_TF:
        particles_init_potential_tf(p, pot_params);
        break;
    case PARTICLE_POTENTIAL_TYPE_LJ:
        particles_init_potential_lj(p, pot_params);
        break;
    case PARTICLE_POTENTIAL_TYPE_SC:
        particles_init_potential_sc(p, pot_params);
        break;
    default:
        mpi_fprintf(stderr, "Invalid potential type %d\n", pot_type);
        exit(1);
        break;
    }
}

static void particles_init_pairwise_dielectric(
    particles *p, double eps_s, double eps_int, double *radii, int model
) {
    if (p->pairwise_dielectric_model != PAIRWISE_DIELECTRIC_MODEL_NONE) {
        mpi_fprintf(stderr, "A pairwise dielectric model is already initialized.\n");
        exit(1);
    }
    p->pairwise_dielectric_radii = (double *)malloc(p->n_p * sizeof(double));
    if (p->pairwise_dielectric_radii == NULL) {
        mpi_fprintf(stderr, "Could not allocate pairwise dielectric radii.\n");
        exit(1);
    }
    memcpy(p->pairwise_dielectric_radii, radii, p->n_p * sizeof(double));
    p->pairwise_dielectric_eps_s = eps_s;
    p->pairwise_dielectric_eps_int = eps_int;
    p->pairwise_dielectric_model = model;
}

void particles_init_sphere_pairwise_harmonic(
    particles *p, double eps_s, double eps_int, double *radii
) {
    particles_init_pairwise_dielectric(
        p, eps_s, eps_int, radii,
        PAIRWISE_DIELECTRIC_MODEL_SPHERE_HARMONIC
    );
}

void particles_init_ribar_window_pairwise(
    particles *p, double eps_s, double eps_int, double window, double *radii
) {
    particles_init_pairwise_dielectric(
        p, eps_s, eps_int, radii,
        PAIRWISE_DIELECTRIC_MODEL_RIBAR_WINDOW
    );
    p->ribar_pairwise_window = window;
}

void particles_init_potential_tf(particles *p, double *pot_params) {
    int typ1, typ2;
    int n_p = p->n_p;
    int n_typ = p->n_typ;
    long int np2 = n_p * n_p;

    p->tf_params = (double *)malloc(7 * np2 * sizeof(double));

    double r_cut = p->L / 2.0;
    p->r_cut = r_cut;
    double r_cut_6 = pow(r_cut, 6);
    double r_cut_7 = r_cut_6 * r_cut;
    double r_cut_8 = r_cut_7 * r_cut;
    double r_cut_9 = r_cut_8 * r_cut;

    long int in, inj, idx;
    double A, B, C, D, sigma, v_shift, alpha, beta;
    for (int i = 0; i < n_p; i++) {
        in = i * n_p;
        typ1 = p->types[i];
        for (int j = 0; j < n_p; j++) {
            inj = in + j;
            typ2 = p->types[j];

            idx = (typ1 * n_typ + typ2) * 5;  // Assuming pot_params is structured as [A, B, C, D, sigma] for each type pair

            A = pot_params[idx + 0];
            B = pot_params[idx + 1];
            C = pot_params[idx + 2];
            D = pot_params[idx + 3];
            sigma = pot_params[idx + 4];

            v_shift = A * exp(B * (sigma - r_cut)) - C / r_cut_6 - D / r_cut_8;
            alpha = A * B * exp(B * (sigma - r_cut)) - 6 * C / r_cut_7 - 8 * D / r_cut_9;
            beta = - v_shift - alpha * r_cut;

            p->tf_params[0*np2 + inj] = A;
            p->tf_params[1*np2 + inj] = B;
            p->tf_params[2*np2 + inj] = C;
            p->tf_params[3*np2 + inj] = D;
            p->tf_params[4*np2 + inj] = sigma;
            p->tf_params[5*np2 + inj] = alpha;
            p->tf_params[6*np2 + inj] = beta;
        }
    }

    p->compute_forces_noel = particles_compute_forces_tf;
}

void particles_init_potential_lj(particles *p, double *pot_params) {
    int typ1, typ2;
    int n_p = p->n_p;
    int n_typ = p->n_typ;
    long int np2 = n_p * n_p;

    p->lj_params = (double *)malloc(4 * np2 * sizeof(double));

    double r_cut = p->L / 2.0;
    p->r_cut = r_cut;
    
    long int in, inj, idx;
    double sigma, epsilon, v_shift, alpha, beta;
    for (int i = 0; i < n_p; i++) {
        in = i * n_p;
        typ1 = p->types[i];
        for (int j = 0; j < n_p; j++) {
            inj = in + j;
            typ2 = p->types[j];

            idx = (typ1 * n_typ + typ2) * 2;  // Input pot_params is structured as [sigma, epsilon] for each type pair

            sigma = pot_params[idx + 0];
            epsilon = pot_params[idx + 1];

            v_shift = 4 * epsilon * (pow(sigma / r_cut, 12) - pow(sigma / r_cut, 6));
            alpha = 4 * (12 * epsilon * pow(sigma / r_cut, 12) - 6 * epsilon * pow(sigma / r_cut, 6)) / r_cut;
            beta = - v_shift - alpha * r_cut;

            p->lj_params[0*np2 + inj] = sigma;
            p->lj_params[1*np2 + inj] = epsilon;
            p->lj_params[2*np2 + inj] = alpha;
            p->lj_params[3*np2 + inj] = beta;
        }
    }

    p->compute_forces_noel = particles_compute_forces_lj;
}

void particles_init_potential_sc(particles *p, double *pot_params) {
    const int n_typ = p->n_typ;
    const long int n_pairs = (long int)n_typ * n_typ;
    p->sc_params = (double *)malloc(5 * n_pairs * sizeof(double));
    if (p->sc_params == NULL) {
        mpi_fprintf(stderr, "Could not allocate pair-specific SC parameters.\n");
        exit(1);
    }

    p->r_cut = 0.5 * p->L;
    const double r_cut = p->r_cut;
    for (long int pair = 0; pair < n_pairs; pair++) {
        const long int input_idx = 3 * pair;
        const long int output_idx = 5 * pair;
        const double nu = pot_params[input_idx];
        const double d = pot_params[input_idx + 1];
        const double B_nu = pot_params[input_idx + 2];
        const double d_over_r_cut_pow = pow(d / r_cut, nu);
        const double alpha = B_nu * nu * d_over_r_cut_pow / r_cut;
        const double beta = -B_nu * d_over_r_cut_pow - alpha * r_cut;

        p->sc_params[output_idx] = nu;
        p->sc_params[output_idx + 1] = d;
        p->sc_params[output_idx + 2] = B_nu;
        p->sc_params[output_idx + 3] = alpha;
        p->sc_params[output_idx + 4] = beta;
    }
    p->compute_forces_noel = particles_compute_forces_sc;
}

void particles_update_nearest_neighbors_cic(particles *p) {
    int np = p->n_p;
    int n = p->n;
    double h = p->h;
    double L = p->L;

    long int *neighbors = p->neighbors;
    double *pos = p->pos;

    int i, j;
    long int i0, i1;
    int ni, nj, nk, nip, njp, nkp;

    #pragma omp parallel for private(i, j, i0, i1, ni, nj, nk, nip, njp, nkp)
    for (i = 0; i < np; i++) {
        i0 = i * 3;
        i1 = i * NUM_NEIGH_CIC * 3;

        ni = (int)floor(pos[i0] / h);
        nj = (int)floor(pos[i0 + 1] / h);
        nk = (int)floor(pos[i0 + 2] / h);

        nip = (ni + 1) % n;
        njp = (nj + 1) % n;
        nkp = (nk + 1) % n;

        for (int j = 0; j < 24; j += 3) {
            neighbors[i1 + j + 0] = ni;
            neighbors[i1 + j + 1] = nj;
            neighbors[i1 + j + 2] = nk;
        }

        neighbors[i1 +  3 + 0] = nip;  // 1,0,0
        neighbors[i1 +  6 + 1] = njp;  // 0,1,0
        neighbors[i1 +  9 + 2] = nkp;  // 0,0,1

        neighbors[i1 + 12 + 0] = nip;  // 1,1,0
        neighbors[i1 + 12 + 1] = njp;

        neighbors[i1 + 15 + 0] = nip;  // 1,0,1
        neighbors[i1 + 15 + 2] = nkp;

        neighbors[i1 + 18 + 1] = njp;  // 0,1,1
        neighbors[i1 + 18 + 2] = nkp;

        neighbors[i1 + 21 + 0] = nip;  // 1,1,1
        neighbors[i1 + 21 + 1] = njp;
        neighbors[i1 + 21 + 2] = nkp;
    }
}

void particles_update_nearest_neighbors_spline(particles *p) {    
    int np = p->n_p;
    int n = p->n;
    double h = p->h;
    double L = p->L;

    long int *neighbors = p->neighbors;
    double *pos = p->pos;

    int i;
    long int i0, i1;
    int ni, nj, nk, nip, njp, nkp;

    #pragma omp parallel for private(i, i0, i1, ni, nj, nk, nip, njp, nkp)
    for (i = 0; i < np; i++) {
        i0 = i * 3;
        i1 = i * NUM_NEIGH_SPLINE * 3;

        ni = (int)floor(pos[i0] / h);
        nj = (int)floor(pos[i0 + 1] / h);
        nk = (int)floor(pos[i0 + 2] / h);

        for (int a=-1; a <= 2; a++) {
            nip = (ni + a + n) % n;
            for (int b=-1; b <= 2; b++) {
                njp = (nj + b + n) % n;
                for (int c=-1; c <= 2; c++) {
                    nkp = (nk + c + n) % n;

                    neighbors[i1] = nip;
                    neighbors[i1 + 1] = njp;
                    neighbors[i1 + 2] = nkp;
                    i1 += 3;
                }
            }
        }
    }    
}

double particles_compute_forces_field(particles *p, grid *grid) {
    if (grid->pb_enabled && grid->pb_force_type == PB_FORCE_TYPE_STRESS_TENSOR) {
        // The tensor forces are already resetting this 2 zero, not sure which is the best place to do this
        // depends on who we want to be responsible for what
        // memset(p->fcs_elec, 0, p->n_p * 3 * sizeof(double));
        return 0.0;
    }
    return compute_force_fd(
        p->n, p->n_p, p->h, p->num_neighbors,
        grid->phi_n, p->neighbors, p->charges, p->pos, p->fcs_elec,
        p->charges_spread_func
    );
}

double particles_compute_forces_tf(particles *p) {
    return compute_tf_forces(p->n_p, p->L, p->pos, p->tf_params, p->r_cut, p->fcs_noel);
}

double particles_compute_forces_sc(particles *p) {
    return compute_sc_forces_pairwise(
        p->n_p, p->n_typ, p->L, p->types, p->pos,
        p->sc_params, p->r_cut, p->fcs_noel
    );
}

double particles_compute_forces_lj(particles *p) { 
    return compute_lj_forces(p->n_p, p->L, p->pos, p->lj_params, p->r_cut, p->fcs_noel);
}

double particles_compute_forces_sphere_pairwise_harmonic(particles *p) {
    if (p->pairwise_dielectric_model != PAIRWISE_DIELECTRIC_MODEL_SPHERE_HARMONIC) {
        return 0.0;
    }
    return compute_sphere_pairwise_harmonic_correction(
        p->n_p, p->L, p->h, p->pairwise_dielectric_eps_int,
        p->pairwise_dielectric_eps_s, p->charges, p->pos,
        p->pairwise_dielectric_radii, p->fcs_elec
    );
}

double particles_compute_forces_ribar_window_pairwise(particles *p) {
    if (p->pairwise_dielectric_model != PAIRWISE_DIELECTRIC_MODEL_RIBAR_WINDOW) {
        return 0.0;
    }
    return compute_ribar_window_pairwise_correction(
        p->n_p, p->L, p->ribar_pairwise_window,
        p->pairwise_dielectric_eps_int, p->pairwise_dielectric_eps_s,
        p->charges, p->pos, p->pairwise_dielectric_radii, p->fcs_elec
    );
}

double calc_h_ratio(double rad, double w2, double w3) {
    return (
        (
            -(3 / (4 * w3)) * pow(rad, 2) +
             (3 / (2 * w2)) * rad
        ) /
        (
            -(1 / (4 * w3)) * pow(rad, 3) +
             (3 / (4 * w2)) * pow(rad, 2)
        ) 
    );
}

double particles_compute_forces_pb_roux(particles *p, grid *g) {
    if (! g->pb_enabled) {
        // Poisson-Boltzmann is not enabled
        mpi_fprintf(stderr, "Poisson-Boltzmann forces are not enabled in the grid.\n");
        exit(1);
    }
    int n = g->n;
    int n_local = g->n_local;
    int n_start = g->n_start;

    double h = g->h;
    double L = g->L;
    double w = g->w;

    double eps_s = g->eps_s;
    double eps_int = g->eps_int;
    double kbar2 = g->kbar2;
    double r_solv;

    long int n2 = n * n;

    double px, py, pz;
    int idx_x, idx_y, idx_z;

    double w2 = w * w;  // Square of the ionic boundary width
    double w3 = w2 * w;  // Cube of the ionic boundary width
    double hd2 = h / 2.0;  // Half the grid spacing
    double h3 = h * h * h;  // Cube of the grid spacing

    // double S = 0.0;  // Sum of the derivatives of the dielectric constant
    double non_polar_energy = 0.0;

    long int size = p->n_p * 3;
    double *fcs_db = p->fcs_db;
    double *fcs_ib = p->fcs_ib;
    double *fcs_np = p->fcs_np;

    memset(fcs_db, 0, size * sizeof(double));  // Initialize forces to zero
    memset(fcs_ib, 0, size * sizeof(double));  // Initialize forces to zero
    memset(fcs_np, 0, size * sizeof(double));  // Initialize non-polar forces to zero

    mpi_grid_exchange_bot_top(g->phi_n, n_local, n);
    mpi_grid_exchange_bot_top(g->eps_x, n_local, n);
    mpi_grid_exchange_bot_top(g->eps_y, n_local, n);
    mpi_grid_exchange_bot_top(g->eps_z, n_local, n);

    #pragma \
        omp parallel for private(r_solv, px, py, pz, idx_x, idx_y, idx_z) \
        reduction(+:non_polar_energy, fcs_db[:size], fcs_ib[:size], fcs_np[:size])
    for (int np = 0; np < p->n_p; np++) {
        int np3 = np * 3;

        r_solv = p->solv_radii[np];
        px = p->pos[np3];
        py = p->pos[np3 + 1];
        pz = p->pos[np3 + 2];

        double r1, r2;
        double r_solv_p2 = pow(r_solv + w, 2);
        double r_solv_m2 = pow(r_solv - w, 2);

        int idx_range = (int)floor((r_solv + w) / h) + 2;

        idx_x = (int)floor(px / h);
        idx_y = (int)floor(py / h);
        idx_z = (int)floor(pz / h);

        double dx, dy, dz;
        double dx2, dy2, dz2;
        double rx, ry, rz;  // Relative coordinates
        double app1, app2;

        int i0, j0, k0, i1, i2, j1, j2, k1, k2;
        long int idx_cen;
        long int idx_bwd_x, idx_bwd_y, idx_bwd_z;
        long int idx_fwd_x, idx_fwd_y, idx_fwd_z;
        
        double phi_center, phi_bwd, phi_fwd, delta_phi;
        double h_ratio;

        double d_eps_x, d_eps_y, d_eps_z, d_eps_norm, inv_grad;

        double eps_x_cen, eps_y_cen, eps_z_cen;
        double eps_x_bwd, eps_y_bwd, eps_z_bwd;

        for (int di = -idx_range; di <= idx_range; di++) {
            i0 = idx_x + di;
            dx = px - i0 * h;  // Calculate the distance in x direction
            dx2 = dx * dx;
            i0 = ((i0 + n) % n);  // Wrap around for periodic boundary conditions
            i0 -= n_start;  // Adjust for local grid start
            if (i0 < 0 || i0 >= n_local) { 
                continue;  // Skip if the point is outside the local grid
            }
            i0 *= n2;  // Convert to linear index
            i1 = i0 + n2;
            i2 = i0 - n2;
            for (int dj = -idx_range; dj <= idx_range; dj++) {
                j0 = idx_y + dj;
                dy = py - j0 * h;  // Calculate the distance in y direction
                dy2 = dy * dy;
                j0 = (j0 + n) % n;  // Wrap around for periodic boundary conditions
                j1 = ((j0 + 1) % n) * n;
                j2 = ((j0 - 1 + n) % n) * n;  // Wrap around for periodic boundary conditions
                j0 *= n;
                for (int dk = -idx_range; dk <= idx_range; dk++) {
                    k0 = idx_z + dk;
                    dz = pz - k0 * h;  // Calculate the distance in z direction
                    dz2 = dz * dz;
                    k0 = (k0 + n) % n;  // Wrap around for periodic boundary conditions
                    k1 = (k0 + 1) % n;
                    k2 = (k0 - 1 + n) % n;  // Wrap around for periodic boundary conditions

                    idx_cen   = i0 + j0 + k0;  // Calculate the index in the grid
                    idx_fwd_x = i1 + j0 + k0;
                    idx_fwd_y = i0 + j1 + k0;
                    idx_fwd_z = i0 + j0 + k1;
                    idx_bwd_x = i2 + j0 + k0;
                    idx_bwd_y = i0 + j2 + k0;
                    idx_bwd_z = i0 + j0 + k2;

                    eps_x_cen = g->eps_x[idx_cen];
                    eps_y_cen = g->eps_y[idx_cen];
                    eps_z_cen = g->eps_z[idx_cen];
                    eps_x_bwd = g->eps_x[idx_bwd_x];
                    eps_y_bwd = g->eps_y[idx_bwd_y];
                    eps_z_bwd = g->eps_z[idx_bwd_z];

                    d_eps_x = eps_x_cen - eps_x_bwd;
                    d_eps_y = eps_y_cen - eps_y_bwd;
                    d_eps_z = eps_z_cen - eps_z_bwd;
                    d_eps_norm = sqrt(d_eps_x * d_eps_x + d_eps_y * d_eps_y + d_eps_z * d_eps_z) / h;
                    // S += d_eps_norm;
                    inv_grad = (d_eps_norm > 0.0) ? 1.0 / d_eps_norm : 0.0;

                    phi_center = g->phi_n[idx_cen];

                    r2 = dx2 + dy2 + dz2;
                    // *************** CENTER***************
                    if (r2 >= r_solv_p2) {
                        // Outside the radius, do nothing
                    } else if (r2 > r_solv_m2) {
                        // Inside the transition region, set dielectric constant to a fraction
                        r1 = sqrt(r2) + 1E-12;
                        rx = dx / r1;
                        ry = dy / r1;
                        rz = dz / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        // *********************** Ionic boundary forces ***********************
                        app1 = g->k2[idx_cen] * phi_center * phi_center * h_ratio;
                        fcs_ib[np3    ] += app1 * rx;
                        fcs_ib[np3 + 1] += app1 * ry;
                        fcs_ib[np3 + 2] += app1 * rz;

                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** X - h/2 ***************
                    app1 = dx + hd2;  // Adjust for half the grid spacing
                    r2 = app1 * app1 + dy2 + dz2;
                    app2 = inv_grad * d_eps_x;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = app1 / r1;
                        ry = dy / r1;
                        rz = dz / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_x_bwd - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_bwd_x] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] -= app1 * app2 * rx;
                            fcs_np[np3 + 1] -= app1 * app2 * ry;
                            fcs_np[np3 + 2] -= app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** X + h/2 ***************
                    app1 = dx - hd2;  // Adjust for half the grid spacing
                    r2 = app1 * app1 + dy2 + dz2;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = app1 / r1;
                        ry = dy / r1;
                        rz = dz / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_x_cen - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_fwd_x] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] += app1 * app2 * rx;
                            fcs_np[np3 + 1] += app1 * app2 * ry;
                            fcs_np[np3 + 2] += app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** Y - h/2 ***************
                    app1 = dy + hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + app1 * app1 + dz2;
                    app2 = inv_grad * d_eps_y;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = dx / r1;
                        ry = app1 / r1;
                        rz = dz / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_y_bwd - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_bwd_y] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] -= app1 * app2 * rx;
                            fcs_np[np3 + 1] -= app1 * app2 * ry;
                            fcs_np[np3 + 2] -= app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** Y + h/2 ***************
                    app1 = dy - hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + app1 * app1 + dz2;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = dx / r1;
                        ry = app1 / r1;
                        rz = dz / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_y_cen - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_fwd_y] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] += app1 * app2 * rx;
                            fcs_np[np3 + 1] += app1 * app2 * ry;
                            fcs_np[np3 + 2] += app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** Z - h/2 ***************
                    app1 = dz + hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + dy2 + app1 * app1;
                    app2 = inv_grad * d_eps_z;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = dx / r1;
                        ry = dy / r1;
                        rz = app1 / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_z_bwd - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_bwd_z] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] -= app1 * app2 * rx;
                            fcs_np[np3 + 1] -= app1 * app2 * ry;
                            fcs_np[np3 + 2] -= app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }

                    // *************** Z + h/2 ***************
                    app1 = dz - hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + dy2 + app1 * app1;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        r1 = sqrt(r2) + 1E-12;
                        rx = dx / r1;
                        ry = dy / r1;
                        rz = app1 / r1;
                        h_ratio = calc_h_ratio(r1 - r_solv + w, w2, w3);

                        app1 = (eps_z_cen - eps_int) * h_ratio;
                        // *********************** Dielectric boundary forces ***********************
                        delta_phi = g->phi_n[idx_fwd_z] - phi_center;
                        fcs_db[np3    ] -= app1 * delta_phi * phi_center * rx;
                        fcs_db[np3 + 1] -= app1 * delta_phi * phi_center * ry;
                        fcs_db[np3 + 2] -= app1 * delta_phi * phi_center * rz;

                        if (g->nonpolar_enabled){
                            // *********************** Non-polar forces ***********************
                            fcs_np[np3    ] += app1 * app2 * rx;
                            fcs_np[np3 + 1] += app1 * app2 * ry;
                            fcs_np[np3 + 2] += app1 * app2 * rz;
                        }
                    } else {
                        // Inside the radius, do nothing
                    }
                }
            }
        }
    }

    allreduce_sum(fcs_db, size);
    allreduce_sum(fcs_ib, size);
    allreduce_sum(fcs_np, size);

    dscal(fcs_db, h / (8.0 * M_PI), size);
    dscal(fcs_ib, h / (8.0 * M_PI), size);

    if (g->nonpolar_enabled && fabs(p->gamma_np) > 0.0) {
        double eps_delta = eps_s - eps_int;
        if (fabs(eps_delta) < 1e-12) {
            mpi_fprintf(
                stderr,
                "Invalid Poisson-Boltzmann setup: nonpolar forces require eps_s != eps_int.\n"
            );
            exit(1);
        }
        dscal(fcs_np, -p->gamma_np * h / eps_delta, size);
    }

    return non_polar_energy;
}

double particles_compute_forces_pb_stress_tensor(particles *p, grid *g) {
    long int size = p->n_p * 3;
    double non_polar_energy = 0.0;

    if (g->nonpolar_enabled) {
        // Call the Roux method directly to avoid infinite recursion:
        // particles_compute_forces_pb would re-dispatch to this function.
        non_polar_energy = particles_compute_forces_pb_roux(p, g);
    } else {
        memset(p->fcs_np, 0, size * sizeof(double));
    }

    compute_stress_tensor_forces(
        g->n, g->eps_s, p->n_p, g->L, g->h,
        g->phi_n, g->region, g->st_owner, p->pos, p->solv_radii, p->fcs_elec,
        g->stress_tensor_bc_type == STRESS_TENSOR_BC_TYPE_PBC
    );

    memset(p->fcs_db, 0, size * sizeof(double));
    memset(p->fcs_ib, 0, size * sizeof(double));
    return non_polar_energy;
}

double particles_compute_forces_pb(particles *p, grid *g) {
    switch (g->pb_force_type)
    {
        case MAP_NOT_INITIALIZED:
            mpi_fprintf(stderr, "Poisson-Boltzmann force type not initialized. Please call `grid_pb_init` before running Poisson-Boltzmann related functions.\n");
            exit(1);
        case PB_FORCE_TYPE_PB_ROUX:
            return particles_compute_forces_pb_roux(p, g);
        case PB_FORCE_TYPE_STRESS_TENSOR:
            return particles_compute_forces_pb_stress_tensor(p, g);
        default:
            mpi_fprintf(stderr, "Invalid Poisson-Boltzmann force type specified.\n");
            exit(1);
    }
}

void particles_compute_forces_tot(particles *p) {
    int size = p->n_p * 3;
    memset(p->fcs_tot, 0, size * sizeof(double));  // Initialize total forces to zero
    if (p->fcs_elec != NULL) {
        daxpy(p->fcs_elec, p->fcs_tot, 1.0, size);  // Add the electric forces
    }
    if (p->fcs_noel != NULL) {
        daxpy(p->fcs_noel, p->fcs_tot, 1.0, size);
    }
    if (p->fcs_db != NULL) {
        daxpy(p->fcs_db, p->fcs_tot, 1.0, size);
    }
    if (p->fcs_ib != NULL) {
        daxpy(p->fcs_ib, p->fcs_tot, 1.0, size);
    }
    if (p->fcs_np != NULL) {
        daxpy(p->fcs_np, p->fcs_tot, 1.0, size);
    }
} 


double particles_get_temperature(particles *p) {
    return 2 * particles_get_kinetic_energy(p) / (3 * p->n_p * kB);
}

double particles_get_kinetic_energy(particles *p) {
    long int ni;
    double kin = 0.0;
    double app;

    #pragma omp parallel for private(ni, app) reduction(+:kin)
    for (int i = 0; i < p->n_p; i++) {
        ni = i * 3;
        app = 0.0;
        // printf("vel: %e, %e, %e\n", p->vel[ni], p->vel[ni + 1], p->vel[ni + 2]);
        for (int j = 0; j < 3; j++) {
            app += pow(p->vel[ni + j], 2);
        }
        // printf("app: %e, mass: %f\n", app, p->mass[i]);
        kin += p->mass[i] * app;
    }

    return 0.5 * kin;
}

void particles_get_momentum(particles *p, double *out) {
    int ni;
    double mass;
    double px = 0.0, py = 0.0, pz = 0.0;

    #pragma omp parallel for private(ni, mass) reduction(+:px, py, pz)
    for (int i = 0; i < p->n_p; i++) {
        ni = i * 3;
        mass = p->mass[i];
        px += mass * p->vel[ni];
        py += mass * p->vel[ni + 1];
        pz += mass * p->vel[ni + 2];
    }

    out[0] = px;
    out[1] = py;
    out[2] = pz;
}

void particles_rescale_velocities(particles *p) {
    double *init_vel = (double *)calloc(p->n_typ * 3, sizeof(double));

    for (int i = 0; i < p->n_p; i++) {
        for (int j = 0; j < 3; j++) {
            init_vel[p->types[i] * 3 + j] += p->vel[i * 3 + j];
        }
    }

    for (int i = 0; i < p->n_p; i++) {
        for (int j = 0; j < 3; j++) {
            p->vel[i * 3 + j] -= 2 * init_vel[p->types[i] * 3 + j] / p->n_p;
        }
    }

    free(init_vel);
}
