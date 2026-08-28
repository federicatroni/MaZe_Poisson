#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "sphere_intersect.h"
#include "mpi_base.h"
#include "mp_structs.h"
#include "linalg.h"

static int pbc_grid_index(int idx, int n) {
    // Same periodic index wrap used elsewhere as (idx + n) % n, generalized for larger offsets.
    idx %= n;
    if (idx < 0) idx += n;
    return idx;
}

static long grid_index_3d(int i, int j, int k, int n) {
    return (long)k + (long)j * n + (long)i * n * n;
}

// /*
// Compute the forces on each particle by computing the field from the potential using finite differences.
// New version computes the field only where the particles are located.

// @param n_grid: the number of grid points in each dimension
// @param n_p: the number of particles
// @param h: the grid spacing
// @param num_neigh: the number of neighbors for each particle
// @param phi: the potential field of size n_grid * n_grid * n_grid
// @param neighbors: Array (x,y,z) of neighbors indexes for each particle (n_p x 8 x 3)
// @param charges: the charges on each particle of size n_p
// @param pos: the positions of the particles of size n_p * 3
// @param forces: the output forces on each particle of size n_p * 3
// @param g: the function to compute the charge assignment

// @return the sum of the charges on the neighbors
// */
double compute_force_fd(
    int n_grid, int n_p, double h, int num_neigh,
    double *phi, long int *neighbors, double *charges, double *pos, double *forces,
    double (*g)(double, double, double)
) {
    int nn3 = num_neigh * 3;
    long int n = n_grid;
    long int n2 = n * n;

    long int i, j, k, jn, in2;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k0, k1, k2;
    double E, qc;

    int n_loc = get_n_loc();
    int n_start = get_n_start();

    double const h2 = 2.0 * h;
    double const L = n * h;
    double px, py, pz, chg;
    
    // Exchange the top and bottom slices
    mpi_grid_exchange_bot_top(phi, n_loc, n);

    double sum_q = 0.0;
    #pragma omp parallel for private(i, j, k, i0, i1, i2, in2, j0, j1, j2, jn, k0, k1, k2, E, qc, px, py, pz, chg) reduction(+:sum_q)
    for (int ip = 0; ip < n_p; ip++) {
        i0 = ip * nn3;
        j0 = ip*3;
        forces[j0] = 0.0;
        forces[j0+1] = 0.0;
        forces[j0+2] = 0.0;
        px = pos[j0];
        py = pos[j0 + 1];
        pz = pos[j0 + 2];
        chg = charges[ip];
        // printf("ip: %d, chg: %f, px: %f, py: %f, pz: %f L: %f, h: %f\n", ip, chg, px, py, pz, L, h);
        for (int in = 0; in < nn3; in += 3) {
            i1 = i0 + in;
            i = neighbors[i1] - n_start;
            if (i < 0 || i >= n_loc) {
                continue;
            }
            j = neighbors[i1 + 1];
            k = neighbors[i1 + 2];

            in2 = i * n2;
            jn = j * n;

            qc = chg * g(px - (i+n_start)*h, L, h) * g(py - j*h, L, h) * g(pz - k*h, L, h);
            sum_q += qc;
            // X
            i1 = (i+1) * n2;
            i2 = (i-1) * n2;
            E = (phi[i2 + jn + k] - phi[i1 + jn + k]) / h2;
            forces[j0] += qc * E;
            // Y
            j1 = ((j+1) % n) * n;
            j2 = ((j-1 + n) % n) * n;
            E = (phi[in2 + j2 + k] - phi[in2 + j1 + k]) / h2;
            forces[j0 + 1] += qc * E;
            // Z
            k1 = ((k+1) % n);
            k2 = ((k-1 + n) % n);
            E = (phi[in2 + jn + k2] - phi[in2 + jn + k1]) / h2;
            forces[j0 + 2] += qc * E;
        }
    }

    allreduce_sum(&sum_q, 1);
    allreduce_sum(forces, 3 * n_p);

    return sum_q;
}

/*
Compute the particle-particle forces using the tabulated Tosi-Fumi potential

@param n_p: the number of particles
@param L: the size of the box
@param pos: the positions of the particles (n_p, 3)
@param params: the parameters of the potential [A, B, C, D, sigma, alpha, beta] (7, n_p, n_p)
@param r_cut: the cutoff radius
@param forces: the output forces on each particle (n_p, 3)
*/
double compute_tf_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces) {
    int ip, jp;
    int n_p2 = 2 * n_p;
    long int n_p_pow2 = n_p * n_p;
    long int idx1, idx2;

    double *A = params;
    double *B = A + n_p_pow2;
    double *C = B + n_p_pow2;
    double *D = C + n_p_pow2;
    double *sigma_TF = D + n_p_pow2;
    double *alpha = sigma_TF + n_p_pow2;
    double *beta = alpha + n_p_pow2;

    double app;
    double r_diff[3];
    double r_mag, f_mag, V_mag;
    double potential_energy = 0.0;
    double a, b, c, d, sigma, al, be;

    #pragma omp parallel for private(app, ip, jp, r_diff, r_mag, f_mag, V_mag, a, b, c, d, sigma, al, be, idx1, idx2) reduction(+:potential_energy)
    for (int i = 0; i < n_p; i++) {
        r_mag = 0.0;
        ip = i * 3;
        idx1 = i * n_p;
        forces[ip] = 0.0;
        forces[ip + 1] = 0.0;
        forces[ip + 2] = 0.0;
        for (int j = 0; j < n_p; j++) {
            if (i == j) {
                continue;
            }
            jp = 3 * j;
            app = pos[ip] - pos[jp];
            app -= L * round(app / L);
            r_mag = app * app;
            r_diff[0] = app;
            app = pos[ip + 1] - pos[jp + 1];
            app -= L * round(app / L);
            r_diff[1] = app;
            r_mag += app * app;
            app = pos[ip + 2] - pos[jp + 2];
            app -= L * round(app / L);
            r_diff[2] = app;
            r_mag += app * app;
            r_mag = sqrt(r_mag);
            if (r_mag > r_cut) {
                continue;
            }
            r_diff[0] /= r_mag;
            r_diff[1] /= r_mag;
            r_diff[2] /= r_mag;
                
            idx2 = idx1 + j;
            a = A[idx2];
            b = B[idx2];
            c = C[idx2];
            d = D[idx2];
            sigma = sigma_TF[idx2];
            al = alpha[idx2];
            be = beta[idx2];

            f_mag = b * a * exp(b * (sigma - r_mag)) - 6 * c / pow(r_mag, 7) - 8 * d / pow(r_mag, 9) - al;
            V_mag = a * exp(b * (sigma - r_mag)) - c / pow(r_mag, 6) - d / pow(r_mag, 8) + al * r_mag + be;

            forces[ip] += f_mag * r_diff[0];
            forces[ip + 1] += f_mag * r_diff[1];
            forces[ip + 2] += f_mag * r_diff[2];

            potential_energy += V_mag;
        }
    }

    return potential_energy / 2;
}


/*
Compute the particle-particle forces using the tabulated Lennard-Jones potential

@param n_p: the number of particles
@param L: the size of the box
@param pos: the positions of the particles (n_p, 3)
@param params: the parameters of the potential [sigma, epsilon] (4, n_p, n_p)
@param r_cut: the cutoff radius
@param forces: the output forces on each particle (n_p, 3)
*/
double compute_lj_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces) {
    int ip, jp;
    int n_p2 = 2 * n_p;
    long int n_p_pow2 = n_p * n_p;
    long int idx1, idx2;

    double *sigma_lj = params;
    double *epsilon_lj = sigma_lj + n_p_pow2;
    double *alpha = epsilon_lj + n_p_pow2;
    double *beta = alpha + n_p_pow2;

    double app;
    double r_diff[3];
    double r_mag, f_mag, V_mag;
    double potential_energy = 0.0;
    double epsilon, sigma, al, be;

    #pragma omp parallel for private(app, ip, jp, r_diff, r_mag, f_mag, V_mag, epsilon, sigma, al, be, idx1, idx2) reduction(+:potential_energy)
    for (int i = 0; i < n_p; i++) {
        r_mag = 0.0;
        ip = i * 3;
        idx1 = i * n_p;
        forces[ip] = 0.0;
        forces[ip + 1] = 0.0;
        forces[ip + 2] = 0.0;
        for (int j = 0; j < n_p; j++) {
            if (i == j) {
                continue;
            }
            jp = 3 * j;
            app = pos[ip] - pos[jp];
            app -= L * round(app / L);
            r_mag = app * app;
            r_diff[0] = app;
            app = pos[ip + 1] - pos[jp + 1];
            app -= L * round(app / L);
            r_diff[1] = app;
            r_mag += app * app;
            app = pos[ip + 2] - pos[jp + 2];
            app -= L * round(app / L);
            r_diff[2] = app;
            r_mag += app * app;
            r_mag = sqrt(r_mag);
            if (r_mag > r_cut) {
                continue;
            }
            r_diff[0] /= r_mag;
            r_diff[1] /= r_mag;
            r_diff[2] /= r_mag;
                
            idx2 = idx1 + j;
            sigma = sigma_lj[idx2];
            epsilon = epsilon_lj[idx2];
            al = alpha[idx2];
            be = beta[idx2];

            //write f_mag and V_mag for lennard-jones potential
            f_mag = 4 * epsilon * (12 * pow(sigma / r_mag, 12) - 6 * pow(sigma / r_mag, 6)) / r_mag - al;
            V_mag = 4 * epsilon * (pow(sigma / r_mag, 12) - pow(sigma / r_mag, 6)) + al * r_mag + be;

            forces[ip] += f_mag * r_diff[0];
            forces[ip + 1] += f_mag * r_diff[1];
            forces[ip + 2] += f_mag * r_diff[2];

            potential_energy += V_mag;
        }
    }

    return potential_energy / 2;
}

/*
 * Pairwise reduction of the SPHERE dielectric map.  Across a layer of width h
 * centred at contact, f is the fraction inside the ionic spheres and the
 * dielectric is mixed harmonically:
 *     1/eps(r) = f/eps_int + (1-f)/eps_s.
 * The homogeneous 1/eps_s field is already present in forces; this routine
 * adds only q_i q_j f (1/eps_int - 1/eps_s) / r and its exact derivative.
 */
double compute_sphere_pairwise_harmonic_correction(
    int n_p, double L, double h, double eps_int, double eps_s,
    const double *charges, const double *pos, const double *radii, double *forces
) {
    double correction_energy = 0.0;
    const int size = 3 * n_p;
    const double delta_inv_eps = 1.0 / eps_int - 1.0 / eps_s;

    #pragma omp parallel reduction(+:correction_energy, forces[:size])
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < n_p; i++) {
            const int ip = 3 * i;
            for (int j = i + 1; j < n_p; j++) {
                const int jp = 3 * j;
                double dr[3];
                double r2 = 0.0;
                for (int k = 0; k < 3; k++) {
                    dr[k] = pos[ip + k] - pos[jp + k];
                    dr[k] -= L * round(dr[k] / L);
                    r2 += dr[k] * dr[k];
                }
                const double r = sqrt(r2);
                if (r == 0.0) {
                    continue;
                }

                const double contact = radii[i] + radii[j];
                const double lower = contact - 0.5 * h;
                const double upper = contact + 0.5 * h;
                double frac;
                double dfrac_dr;
                if (r <= lower) {
                    frac = 1.0;
                    dfrac_dr = 0.0;
                } else if (r < upper) {
                    frac = (upper - r) / h;
                    dfrac_dr = -1.0 / h;
                } else {
                    continue;
                }

                const double qij = charges[i] * charges[j];
                const double inv_r = 1.0 / r;

                correction_energy += qij * delta_inv_eps * frac * inv_r;
                const double f_mag = qij * delta_inv_eps * (
                    frac * inv_r * inv_r - dfrac_dr * inv_r
                );
                for (int k = 0; k < 3; k++) {
                    const double f_k = f_mag * dr[k] * inv_r;
                    forces[ip + k] += f_k;
                    forces[jp + k] -= f_k;
                }
            }
        }
    }
    return correction_energy;
}

/*
 * Pairwise Ribar dielectric window with pair-specific contact distances.
 * For contact = R_i + R_j and outward width Delta:
 *
 *   eps(r) = eps_int                                      r <= contact
 *          = eps_int + (eps_s-eps_int)(r-contact)/Delta  contact < r < contact+Delta
 *          = eps_s                                        r >= contact+Delta.
 *
 * The homogeneous Coulomb contribution at eps_s is already in forces.  This
 * routine adds the correction q_i q_j [1/eps(r)-1/eps_s]/r and its exact
 * radial derivative.  This follows Ribar et al.'s linear ramp in eps rather
 * than the harmonic (linear-in-1/eps) edge mixing used by SPHERE.
 */
double compute_ribar_window_pairwise_correction(
    int n_p, double L, double window, double eps_int, double eps_s,
    const double *charges, const double *pos, const double *radii, double *forces
) {
    double correction_energy = 0.0;
    const int size = 3 * n_p;
    const double deps_dr_window = (eps_s - eps_int) / window;

    #pragma omp parallel reduction(+:correction_energy, forces[:size])
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < n_p; i++) {
            const int ip = 3 * i;
            for (int j = i + 1; j < n_p; j++) {
                const int jp = 3 * j;
                double dr[3];
                double r2 = 0.0;
                for (int k = 0; k < 3; k++) {
                    dr[k] = pos[ip + k] - pos[jp + k];
                    dr[k] -= L * round(dr[k] / L);
                    r2 += dr[k] * dr[k];
                }
                const double r = sqrt(r2);
                if (r == 0.0) {
                    continue;
                }

                const double contact = radii[i] + radii[j];
                const double upper = contact + window;
                if (r >= upper) {
                    continue;
                }

                double eps_d = eps_int;
                double deps_dr = 0.0;
                if (r > contact) {
                    eps_d += deps_dr_window * (r - contact);
                    deps_dr = deps_dr_window;
                }

                const double qij = charges[i] * charges[j];
                const double inv_r = 1.0 / r;
                const double inv_eps = 1.0 / eps_d;
                const double delta_inv_eps = inv_eps - 1.0 / eps_s;

                correction_energy += qij * delta_inv_eps * inv_r;
                const double f_mag = qij * (
                    delta_inv_eps * inv_r * inv_r
                    + deps_dr * inv_eps * inv_eps * inv_r
                );
                for (int k = 0; k < 3; k++) {
                    const double f_k = f_mag * dr[k] * inv_r;
                    forces[ip + k] += f_k;
                    forces[jp + k] -= f_k;
                }
            }
        }
    }
    return correction_energy;
}


/*
Compute the particle-particle forces using the SC repulsive potential

@param n_p: the number of particles
@param L: the size of the box
@param pos: the positions of the particles (n_p, 3)
@param params: the parameters of the potential [nu, d, B] (3)
@param r_cut: the cutoff radius
@param forces: the output forces on each particle (n_p, 3)
*/
double compute_sc_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces) {
    int i, j, k, ip, jp;
    double nu, d, B_nu, alpha, beta;
    double potential_energy = 0.0;

    int size = n_p * 3;

    double app;
    double r_diff[3];
    double r_mag, f_mag, V_mag;
    double d_over_r_pow;
    double f_k;

    nu    = params[0];
    d     = params[1];
    B_nu  = params[2];
    alpha = params[3];
    beta  = params[4];

    memset(forces, 0, size * sizeof(double));

    #pragma \
        omp parallel private(i, j, k, ip, jp, r_diff, r_mag, f_mag, f_k, V_mag, d_over_r_pow) \
        reduction(+:potential_energy, forces[:size])
    for (i = 0; i < n_p; i++) {
        ip = 3 * i;
        for (j = i + 1; j < n_p; j++) {
            jp = 3 * j;

            r_mag = 0.0;
            for (k = 0; k < 3; k++) {
                app = pos[ip + k] - pos[jp + k];
                app -= L * round(app / L);
                r_mag += app * app;
                r_diff[k] = app;
            }

            r_mag = sqrt(r_mag);
            if (r_mag > r_cut) {
                continue;
            }

            d_over_r_pow = pow(d / r_mag, nu);
            V_mag = B_nu * d_over_r_pow + alpha * r_mag + beta;
            f_mag = B_nu * nu * d_over_r_pow / r_mag - alpha;

            for (k = 0; k < 3; k++) {
                f_k = f_mag * r_diff[k] / r_mag;
                forces[ip + k] += f_k;
                forces[jp + k] -= f_k;
            }

            potential_energy += V_mag;
        }
    }
    return potential_energy;
}


/*
Compute the stress tensor forces on particles

@param g: the grid structure containing the grid parameters
@param p: the particles structure containing the particle parameters
@param phi: the potential field of size n_grid * n_grid * n_grid
@param out_forces: the output forces on each particle of size n_p * 3
*/
void compute_stress_tensor_forces_dbc(
    int n, double eps_s, int n_p,
    double L, double h, const double *phi, const unsigned int *region,
    double *pos, double *solv_radii, double *out_forces
) {
    mpi_fprintf(stderr, "`compute_stress_tensor_forces_dbc` should not be used YET!!\n");
    exit(1);
    const double stress_prefactor = 1.0 / (4.0 * M_PI);

    double h2 = h * h;

    double Ex, Ey, Ez;

    #pragma omp parallel for private(Ex, Ey, Ez)
    for (int p_idx = 0; p_idx < n_p; p_idx++) {
        int ip = round(pos[p_idx * 3 + 0] / h);
        int jp = round(pos[p_idx * 3 + 1] / h);
        int kp = round(pos[p_idx * 3 + 2] / h);
        int num_points_min = ceil(solv_radii[p_idx] / h) + 1;
        // +1 rispetto al cubo: garantisce che idx_a sia sempre nel solvente anche quando
        // il centro della particella è sfasato di 0.5*h dal punto di griglia più vicino.
        // Con R = num_points_min la faccia assiale più vicina è a (R-1.5)*h dal centro reale
        // (caso peggiore), potenzialmente dentro la molecola; con R+1 la distanza minima
        // sale a (R-0.5)*h = (num_points_min+0.5)*h >= solv_radii + 0.5*h.
        int R2 = (num_points_min + 1) * (num_points_min + 1);

        for (int di = -num_points_min; di <= num_points_min; di++) {
            for (int dj = -num_points_min; dj <= num_points_min; dj++) {
                for (int dk = -num_points_min; dk <= num_points_min; dk++) {
                    if (di*di + dj*dj + dk*dk >= R2) continue;

                    int i = ip + di, j = jp + dj, k = kp + dk;
                    if (i < 1 || i >= n-1 || j < 1 || j >= n-1 || k < 1 || k >= n-1) continue;

                    long idx_a = (long)k + (long)j * n + (long)i * n * n;
                    // Multi-particella: salta se idx_a è dentro la regione molecolare
                    // (evita di usare eps_s in una zona con eps diversa).
                    if (region != NULL && region[idx_a] != 0) continue;

                    long idx_b;

                    // +x face
                    if ((di+1)*(di+1) + dj*dj + dk*dk >= R2) {
                        idx_b = idx_a + n * n;
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = -(phi[idx_b] - phi[idx_a]) / h;
                            Ey = -((phi[idx_a + n] - phi[idx_a - n]) + (phi[idx_b + n] - phi[idx_b - n])) / (4.0 * h);
                            Ez = -((phi[idx_a + 1] - phi[idx_a - 1]) + (phi[idx_b + 1] - phi[idx_b - 1])) / (4.0 * h);
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * eps_s * (Ex*Ex - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * eps_s * Ex*Ey;
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * eps_s * Ex*Ez;
                        }
                    }

                    // -x face
                    if ((di-1)*(di-1) + dj*dj + dk*dk >= R2) {
                        idx_b = idx_a - n * n;
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = (phi[idx_b] - phi[idx_a]) / h;
                            Ey = -((phi[idx_a + n] - phi[idx_a - n]) + (phi[idx_b + n] - phi[idx_b - n])) / (4.0 * h);
                            Ez = -((phi[idx_a + 1] - phi[idx_a - 1]) + (phi[idx_b + 1] - phi[idx_b - 1])) / (4.0 * h);
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * (-1.0) * eps_s * (Ex*Ex - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * (-1.0) * eps_s * Ex*Ey;
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * (-1.0) * eps_s * Ex*Ez;
                        }
                    }

                    // +y face
                    if (di*di + (dj+1)*(dj+1) + dk*dk >= R2) {
                        idx_b = idx_a + n;
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = -((phi[idx_a + n*n] - phi[idx_a - n*n]) + (phi[idx_b + n*n] - phi[idx_b - n*n])) / (4.0 * h);
                            Ey = -(phi[idx_b] - phi[idx_a]) / h;
                            Ez = -((phi[idx_a + 1] - phi[idx_a - 1]) + (phi[idx_b + 1] - phi[idx_b - 1])) / (4.0 * h);
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * eps_s * Ey*Ex;
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * eps_s * (Ey*Ey - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * eps_s * Ey*Ez;
                        }
                    }

                    // -y face
                    if (di*di + (dj-1)*(dj-1) + dk*dk >= R2) {
                        idx_b = idx_a - n;
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = -((phi[idx_a + n*n] - phi[idx_a - n*n]) + (phi[idx_b + n*n] - phi[idx_b - n*n])) / (4.0 * h);
                            Ey = (phi[idx_b] - phi[idx_a]) / h;
                            Ez = -((phi[idx_a + 1] - phi[idx_a - 1]) + (phi[idx_b + 1] - phi[idx_b - 1])) / (4.0 * h);
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * (-1.0) * eps_s * Ey*Ex;
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * (-1.0) * eps_s * (Ey*Ey - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * (-1.0) * eps_s * Ey*Ez;
                        }
                    }

                    // +z face
                    if (di*di + dj*dj + (dk+1)*(dk+1) >= R2) {
                        idx_b = idx_a + 1;
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = -((phi[idx_a + n*n] - phi[idx_a - n*n]) + (phi[idx_b + n*n] - phi[idx_b - n*n])) / (4.0 * h);
                            Ey = -((phi[idx_a + n] - phi[idx_a - n]) + (phi[idx_b + n] - phi[idx_b - n])) / (4.0 * h);
                            Ez = -(phi[idx_b] - phi[idx_a]) / h;
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * eps_s * Ez*Ex;
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * eps_s * Ez*Ey;
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * eps_s * (Ez*Ez - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                        }
                    }

                    // -z face
                    if (di*di + dj*dj + (dk-1)*(dk-1) >= R2) {
                        idx_b = idx_a - 1;
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = -((phi[idx_a + n*n] - phi[idx_a - n*n]) + (phi[idx_b + n*n] - phi[idx_b - n*n])) / (4.0 * h);
                            Ey = -((phi[idx_a + n] - phi[idx_a - n]) + (phi[idx_b + n] - phi[idx_b - n])) / (4.0 * h);
                            Ez = (phi[idx_b] - phi[idx_a]) / h;
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * (-1.0) * eps_s * Ez*Ex;
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * (-1.0) * eps_s * Ez*Ey;
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * (-1.0) * eps_s * (Ez*Ez - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                        }
                    }
                }
            }
        }
    }
}

void compute_stress_tensor_forces_pbc(
    int n, double eps_s, int n_p,
    double L, double h, double *phi, const unsigned int *region,
    const unsigned int *st_owner,
    double *pos, double *solv_radii, double *out_forces
) {
    long int n2 = n * n;
    const double stress_prefactor = 1.0 / (4.0 * M_PI);
    double h2 = h * h;

    double Ex, Ey, Ez;

    int n_loc = get_n_loc();
    int n_start = get_n_start();
    // Exchange the top and bottom slices
    mpi_grid_exchange_bot_top(phi, n_loc, n);

    #pragma omp parallel for private(Ex, Ey, Ez)
    for (int p_idx = 0; p_idx < n_p; p_idx++) {
        int ip = round(pos[p_idx * 3 + 0] / h);
        int jp = round(pos[p_idx * 3 + 1] / h);
        int kp = round(pos[p_idx * 3 + 2] / h);
        int num_points_min = ceil(solv_radii[p_idx] / h) + 1;
        int R2 = (num_points_min + 1) * (num_points_min + 1);

        int i, j, k;
        int i1, i2, j1, j2, k1, k2;
        long int idx_a;
        long int idx_b;

        long int app1, app2;

        for (int di = -num_points_min; di <= num_points_min; di++) {
            i = pbc_grid_index(ip + di, n) - n_start; // Local index
            i1 = i + 1;
            i2 = i - 1;
            if (i < 0 || i >= n_loc) {
                continue;
            }
            app1 = di * di;
            for (int dj = -num_points_min; dj <= num_points_min; dj++) {
                j  = pbc_grid_index(jp + dj, n);
                j1 = pbc_grid_index(j + 1, n);
                j2 = pbc_grid_index(j - 1, n);
                app2 = app1 + dj * dj;
                for (int dk = -num_points_min; dk <= num_points_min; dk++) {
                    if (app2 + dk*dk >= R2) {
                        continue;
                    }

                    k = pbc_grid_index(kp + dk, n);
                    idx_a = grid_index_3d(i, j, k, n);
                    if (region != NULL && region[idx_a] == 1) {
                        continue;
                    }
                    // Nodes shared by overlapping integration spheres belong to a
                    // single particle, so that no face is summed more than once.
                    // ST_OWNER_NONE means the map was never filled (dielectric
                    // maps other than SPHERE): fall back to the old behaviour.
                    if (st_owner != NULL && st_owner[idx_a] != ST_OWNER_NONE &&
                        st_owner[idx_a] != (unsigned int)p_idx) {
                        continue;
                    }
                    k1 = pbc_grid_index(k + 1, n);
                    k2 = pbc_grid_index(k - 1, n);

                    if ((di+1)*(di+1) + dj*dj + dk*dk >= R2) {
                        idx_b = grid_index_3d(i1, j, k, n);
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = -(phi[idx_b] - phi[idx_a]) / h;
                            Ey = -(
                                (phi[grid_index_3d(i, j1, k, n)] - phi[grid_index_3d(i, j2, k, n)]) +
                                (phi[grid_index_3d(i1, j1, k, n)] - phi[grid_index_3d(i1, j2, k, n)])
                            ) / (4.0 * h);
                            Ez = -(
                                (phi[grid_index_3d(i, j, k1, n)] - phi[grid_index_3d(i, j, k2, n)]) +
                                (phi[grid_index_3d(i1, j, k1, n)] - phi[grid_index_3d(i1, j, k2, n)])
                            ) / (4.0 * h);
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * eps_s * (Ex*Ex - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * eps_s * Ex*Ey;
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * eps_s * Ex*Ez;
                        }
                    }

                    if ((di-1)*(di-1) + dj*dj + dk*dk >= R2) {
                        idx_b = grid_index_3d(i2, j, k, n);
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = (phi[idx_b] - phi[idx_a]) / h;
                            Ey = -(
                                (phi[grid_index_3d(i, j1, k, n)] - phi[grid_index_3d(i, j2, k, n)]) +
                                (phi[grid_index_3d(i2, j1, k, n)] - phi[grid_index_3d(i2, j2, k, n)])
                            ) / (4.0 * h);
                            Ez = -(
                                (phi[grid_index_3d(i, j, k1, n)] - phi[grid_index_3d(i, j, k2, n)]) +
                                (phi[grid_index_3d(i2, j, k1, n)] - phi[grid_index_3d(i2, j, k2, n)])
                            ) / (4.0 * h);
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * (-1.0) * eps_s * (Ex*Ex - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * (-1.0) * eps_s * Ex*Ey;
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * (-1.0) * eps_s * Ex*Ez;
                        }
                    }

                    if (di*di + (dj+1)*(dj+1) + dk*dk >= R2) {
                        idx_b = grid_index_3d(i, j1, k, n);
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = -(
                                (phi[grid_index_3d(i1, j, k, n)] - phi[grid_index_3d(i2, j, k, n)]) +
                                (phi[grid_index_3d(i1, j1, k, n)] - phi[grid_index_3d(i2, j1, k, n)])
                            ) / (4.0 * h);
                            Ey = -(phi[idx_b] - phi[idx_a]) / h;
                            Ez = -(
                                (phi[grid_index_3d(i, j, k1, n)] - phi[grid_index_3d(i, j, k2, n)]) +
                                (phi[grid_index_3d(i, j1, k1, n)] - phi[grid_index_3d(i, j1, k2, n)])
                            ) / (4.0 * h);
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * eps_s * Ey*Ex;
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * eps_s * (Ey*Ey - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * eps_s * Ey*Ez;
                        }
                    }

                    if (di*di + (dj-1)*(dj-1) + dk*dk >= R2) {
                        idx_b = grid_index_3d(i, j2, k, n);
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = -(
                                (phi[grid_index_3d(i1, j, k, n)] - phi[grid_index_3d(i2, j, k, n)]) +
                                (phi[grid_index_3d(i1, j2, k, n)] - phi[grid_index_3d(i2, j2, k, n)])
                            ) / (4.0 * h);
                            Ey = (phi[idx_b] - phi[idx_a]) / h;
                            Ez = -(
                                (phi[grid_index_3d(i, j, k1, n)] - phi[grid_index_3d(i, j, k2, n)]) +
                                (phi[grid_index_3d(i, j2, k1, n)] - phi[grid_index_3d(i, j2, k2, n)])
                            ) / (4.0 * h);
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * (-1.0) * eps_s * Ey*Ex;
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * (-1.0) * eps_s * (Ey*Ey - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * (-1.0) * eps_s * Ey*Ez;
                        }
                    }

                    if (di*di + dj*dj + (dk+1)*(dk+1) >= R2) {
                        idx_b = grid_index_3d(i, j, k1, n);
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = -(
                                (phi[grid_index_3d(i1, j, k, n)] - phi[grid_index_3d(i2, j, k, n)]) +
                                (phi[grid_index_3d(i1, j, k1, n)] - phi[grid_index_3d(i2, j, k1, n)])
                            ) / (4.0 * h);
                            Ey = -(
                                (phi[grid_index_3d(i, j1, k, n)] - phi[grid_index_3d(i, j2, k, n)]) +
                                (phi[grid_index_3d(i, j1, k1, n)] - phi[grid_index_3d(i, j2, k1, n)])
                            ) / (4.0 * h);
                            Ez = -(phi[idx_b] - phi[idx_a]) / h;
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * eps_s * Ez*Ex;
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * eps_s * Ez*Ey;
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * eps_s * (Ez*Ez - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                        }
                    }

                    if (di*di + dj*dj + (dk-1)*(dk-1) >= R2) {
                        idx_b = grid_index_3d(i, j, k2, n);
                        if (region == NULL || region[idx_b] == 0) {
                            Ex = -(
                                (phi[grid_index_3d(i1, j, k, n)] - phi[grid_index_3d(i2, j, k, n)]) +
                                (phi[grid_index_3d(i1, j, k2, n)] - phi[grid_index_3d(i2, j, k2, n)])
                            ) / (4.0 * h);
                            Ey = -(
                                (phi[grid_index_3d(i, j1, k, n)] - phi[grid_index_3d(i, j2, k, n)]) +
                                (phi[grid_index_3d(i, j1, k2, n)] - phi[grid_index_3d(i, j2, k2, n)])
                            ) / (4.0 * h);
                            Ez = (phi[idx_b] - phi[idx_a]) / h;
                            out_forces[p_idx * 3 + 0] += h2 * stress_prefactor * (-1.0) * eps_s * Ez*Ex;
                            out_forces[p_idx * 3 + 1] += h2 * stress_prefactor * (-1.0) * eps_s * Ez*Ey;
                            out_forces[p_idx * 3 + 2] += h2 * stress_prefactor * (-1.0) * eps_s * (Ez*Ez - 0.5*(Ex*Ex + Ey*Ey + Ez*Ez));
                        }
                    }
                }
            }
        }
    }

    allreduce_sum(out_forces, 3 * n_p);
}

void compute_stress_tensor_forces(
    int n, double eps_s, int n_p,
    double L, double h, double *phi, const unsigned int *region,
    const unsigned int *st_owner,
    double *pos, double *solv_radii, double *out_forces, int use_pbc
)
{
    memset(out_forces, 0, n_p * 3 * sizeof(double));

    if (use_pbc) {
        compute_stress_tensor_forces_pbc(n, eps_s, n_p, L, h, phi, region, st_owner, pos, solv_radii, out_forces);
    } else {
        compute_stress_tensor_forces_dbc(n, eps_s, n_p, L, h, phi, region, pos, solv_radii, out_forces);
    }
}
