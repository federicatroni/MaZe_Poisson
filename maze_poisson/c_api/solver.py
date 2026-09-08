import ctypes

import numpy as np
import numpy.ctypeslib as npct

from . import capi

capi.register_function(
    'solver_initialize', None, []
)

# void solverinitialize_grid(int n_grid, double L, double h, double tol, double eps, int grid_type, int precond_type, int y_initial_guess) {
capi.register_function(
    'solver_initialize_grid', None, [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ],
)

# void solver_initialize_grid_pois_boltz(double w, double kbar2, int nonpolar_enabled,
#     int eps_map_type, int pb_force_type, int stress_tensor_bc_type, double kBT, double eps_field_alpha) {
capi.register_function(
    'solver_initialize_grid_pois_boltz', None, [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
    ],
)

# void solver_initialize_grid_smoothing(int method, double r_cut, double sigma, int window_order) {
capi.register_function(
    'solver_initialize_grid_smoothing', None, [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
    ],
)

capi.register_function(
    'solver_set_wendland_poly_coefficients', None, [
        ctypes.c_int,
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ],
)

# void solver_initialize_particles(
#     int n_typ, double L, double h, int n_p, int pot_type, int cas_type,
#     int *types, double *pos, double *vel, double *mass, double *charges,
#     double *pot_params, double r_cut, int lj_force_shift
# ) {
capi.register_function(
    'solver_initialize_particles', None, [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        npct.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_double,
        ctypes.c_int,
    ],
)

# void particles_pb_init(particles *p, double gamma_np, double beta_np, double *solv_radii);
capi.register_function(
    'solver_initialize_particles_pois_boltz', None, [
        ctypes.c_double,
        ctypes.c_double,
        npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ],
)

# void solver_initialize_particles_water(int is_water, int corr_type);
capi.register_function(
    'solver_initialize_particles_water', None, [
        ctypes.c_int,
        ctypes.c_int,
    ],
)

# void solver_initialize_particle_pneigh(int pneigh_method, double r_cut) {
capi.register_function(
    'solver_initialize_particle_pneigh', None, [
        ctypes.c_int,
        ctypes.c_double,
    ],
)

# void solverinitialize_integrator(int n_p, double dt, double T, double gamma, int itg_type, int itg_enabled) {
capi.register_function(
    'solver_initialize_integrator', None, [
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
    ],
)

# void solver_update_particle_neighbors() {
capi.register_function(
    'solver_update_particle_neighbors', None, [],
)

# int solver_update_charges() {
capi.register_function(
    'solver_update_charges', ctypes.c_int, [],
)

# void solver_set_mg_krylov(int val) {
capi.register_function(
    'solver_set_mg_krylov', None, [ctypes.c_int],
    fallback=lambda val: None,
)

# void solver_smoothing() {
capi.register_function(
    'solver_smoothing', None, [],
)

# void solver_init_field() {
capi.register_function(
    'solver_init_field', None, [],
)

# void solver_set_print_convergence(int val) {
capi.register_function(
    'solver_set_print_convergence', None, [
        ctypes.c_int,
    ],
    fallback=lambda val: None,
)

# void solver_set_field(double *phi) {
capi.register_function(
    'solver_set_field', None, [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ],
)

# void solver_set_field_prev(double *phi) {
capi.register_function(
    'solver_set_field_prev', None, [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ],
)

# int solver_update_field() {
capi.register_function(
    'solver_update_field', ctypes.c_int, [],
)

# void solver_update_eps_k2() {
capi.register_function(
    'solver_update_eps_k2', None, [],
)

# double solver_compute_forces_elec() {
capi.register_function(
    'solver_compute_forces_elec', ctypes.c_double, [],
)

# double solver_compute_forces_noel() {
capi.register_function(
    'solver_compute_forces_noel', ctypes.c_double, [],
)

# double solver_compute_forces_pb() {
capi.register_function(
    'solver_compute_forces_pb', ctypes.c_double, [],
)

# double solver_compute_intramolecular_forces() {
capi.register_function(
    'solver_compute_intramolecular_forces', ctypes.c_double, [],
)

# double solver_compute_forces_electrostatic_correction() {
capi.register_function(
    'solver_compute_forces_electrostatic_correction', ctypes.c_double, [],
)

# void solver_compute_forces_tot() {
capi.register_function(
    'solver_compute_forces_tot', None, [],
)

# void integrator_part_1() {
capi.register_function(
    'integrator_part_1', None, [],
)

# void integrator_part_2() {
capi.register_function(
    'integrator_part_2', None, [],
)

# # int solver_nitialize_md(int preconditioning, int vel_rescale) {
# capi.register_function(
#     'solver_initialize_md', ctypes.c_int, [
#         ctypes.c_int,
#         ctypes.c_int,
#     ],
# )

# # void solver_md_loop_iter() {
# capi.register_function(
#     'solver_md_loop_iter', None, [],
# )

# int solver_check_thermostat() {
capi.register_function(
    'solver_check_thermostat', ctypes.c_int, [],
)

# void solver_rescale_velocities() {
capi.register_function(
    'solver_rescale_velocities', None, [],
)

# # void solver_run_n_steps(int n_steps) {
# capi.register_function(
#     'solver_run_n_steps', None, [
#         ctypes.c_int,
#     ],
# )

# void solver_finalize() {
capi.register_function(
    'solver_finalize', None, [],
)

# void set_q(double *q_new) {
capi.register_function(
    'set_q', None, [
        npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'),
    ],
)
