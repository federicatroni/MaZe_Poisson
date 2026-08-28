#ifndef __FORCES_H
#define __FORCES_H

double compute_force_fd(
    int n_grid, int n_p, double h, int num_neigh,
    double *phi, long int *neighbors, double *charges, double *pos, double *forces,
    double (*g)(double, double, double)
);
double compute_tf_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces);
double compute_sc_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces);
double compute_lj_forces(int n_p, double L, double *pos, double *params, double r_cut, double *forces);
double compute_sphere_pairwise_harmonic_correction(
    int n_p, double L, double h, double eps_int, double eps_s,
    const double *charges, const double *pos, const double *radii, double *forces
);
double compute_ribar_window_pairwise_correction(
    int n_p, double L, double window, double eps_int, double eps_s,
    const double *charges, const double *pos, const double *radii, double *forces
);
void compute_stress_tensor_forces_dbc(int n, double eps_s, int n_p, double L, double h, const double *phi, const unsigned int *region, double *pos, double *solv_radii, double *forces);
void compute_stress_tensor_forces_pbc(int n, double eps_s, int n_p, double L, double h, double *phi, const unsigned int *region, const unsigned int *st_owner, double *pos, double *solv_radii, double *forces);
void compute_stress_tensor_forces(int n, double eps_s, int n_p, double L, double h, double *phi, const unsigned int *region, const unsigned int *st_owner, double *pos, double *solv_radii, double *forces, int use_pbc);

#endif
