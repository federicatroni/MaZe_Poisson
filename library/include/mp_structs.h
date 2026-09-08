#ifndef __MP_STRUCTS_H
#define __MP_STRUCTS_H

#define MAP_NOT_INITIALIZED -1
// Max predictor order for the y-history warm start.
#define MAZE_Y_HIST_MAX 2
// Initial-guess modes above MAZE_Y_HIST_MAX are not extrapolation orders.
#define MAZE_Y_GUESS_ZERO (MAZE_Y_HIST_MAX + 1)
#define MAZE_Y_GUESS_MAX MAZE_Y_GUESS_ZERO
#define MAZE_PHI_HIST_MAX 4
#include "enums.h"

#define EPS_MAP_TYPE_NUM 3
#define EPS_MAP_TYPE_TRADITIONAL 0
#define EPS_MAP_TYPE_SPHERE 1
#define EPS_MAP_TYPE_FIELD_DEPENDENT 2

#define PB_FORCE_TYPE_NUM 2
#define PB_FORCE_TYPE_PB_ROUX 0
#define PB_FORCE_TYPE_STRESS_TENSOR 1

#define STRESS_TENSOR_BC_TYPE_NUM 2
#define STRESS_TENSOR_BC_TYPE_DBC 0
#define STRESS_TENSOR_BC_TYPE_PBC 1

// Struct typedefs
typedef struct grid grid;
typedef struct neighbor neighbor;
typedef struct particles particles;
typedef struct integrator integrator;
typedef int y_extrap_order;
typedef int phi_extrap_order;

// Struct function definitions
grid * grid_init(
    int n, double L, double h, double tol, double eps, double eps_int,
    grid_type type, precond_type precond_type, int y_initial_guess, int phi_initial_guess,
    electrostatic_discretization_type discretization, int force_gradient_order
);
neighbor * neighbor_init();
particles * particles_init(int n_p, int n_typ, double L, double h, ca_scheme_type cas_type);
integrator * integrator_init(int n_p, double dt, integrator_type type);

void grid_free(grid *grid);
void neighbor_free(neighbor *n);
void particles_free(particles *p);
void integrator_free(integrator *integrator);

void grid_init_mpi(grid *grid);
void grid_init_mpi_fft(grid *grid);

// long int *neighbor_get_indices(neighbor *n);
// double *neighbor_get_dx(neighbor *n);
// double *neighbor_get_dx(neighbor *n);

void grid_init_mpi(grid *grid);
void grid_init_mpi_fft(grid *grid);

// long int *neighbor_get_indices(neighbor *n);
// double *neighbor_get_dx(neighbor *n);
// double *neighbor_get_dx(neighbor *n);

void grid_pb_init(
    grid *grid, double w, double kbar2, int nonpolar_enabled,
    int eps_map_type, int pb_force_type, int stress_tensor_bc_type, double kBT, double eps_field_alpha
);
void grid_pb_free(grid *grid);
void grid_smoothing_init(grid *grid, int method, double r_cut, double sigma, int window_order);
void grid_deconvolve_window(grid *grid);
void grid_smoothing_free(grid *grid);
void grid_update_eps_and_k2(grid *grid, particles *particles);
double grid_update_eps_field_dependent(grid *grid, particles *particles);
double grid_get_energy_elec(grid *grid);

void lcg_grid_init(grid * grid);
void lcg_grid_cleanup(grid * grid);
void lcg_grid_init_field(grid *grid);
int lcg_grid_update_field(grid *grid);
double lcg_grid_update_charges(grid *grid, particles *p);

void maze_lcg_grid_init(grid * grid);
void maze_lcg_grid_cleanup(grid * grid);
void maze_lcg_grid_init_field(grid *grid);
int maze_lcg_grid_update_field(grid *grid);
double maze_lcg_grid_update_charges(grid *grid, particles *p);

void multigrid_grid_init(grid * grid);
void multigrid_grid_cleanup(grid * grid);
void multigrid_grid_init_field(grid *grid);
int multigrid_grid_update_field(grid *grid);
double multigrid_grid_update_charges(grid *grid, particles *p);

void maze_multigrid_grid_init(grid * grid);
void maze_multigrid_grid_cleanup(grid * grid);
void maze_multigrid_grid_init_field(grid *grid);
int maze_multigrid_grid_update_field(grid *grid);
double maze_multigrid_grid_update_charges(grid *grid, particles *p);

void fft_grid_init(grid * grid);
void fft_grid_cleanup(grid * grid);
void fft_grid_init_field(grid *grid);
int fft_grid_update_field(grid *grid);
double fft_grid_update_charges(grid *grid, particles *p);

void particle_pneigh_init(particles *p, int method, double r_cut);
void particle_pneigh_free(particles *p);

void particles_pb_init(particles *p, double gamma_np, double beta_np, double *solv_radii);
void particles_pb_free(particles *p);

void particles_water_init(particles *p, int is_water, water_electrostatic_type corr_type);
void particles_water_free(particles *p);

void particles_init_potential(particles *p, int pot_type, double *pot_params);
void particles_init_potential_tf(particles *p, double *pot_params);
void particles_init_potential_lj(particles *p, double *pot_params);
void particles_init_potential_sc(particles *p, double *pot_params);
void particles_update_grid_nearest_neighbors_cic(particles *p, grid *g);
void particles_update_grid_nearest_neighbors_spline(particles *p, grid *g);

double particles_compute_forces_field(particles *p, grid *grid);
double particles_compute_forces_tf(particles *p);
double particles_compute_forces_lj(particles *p);
double particles_compute_forces_sc(particles *p);
double particles_compute_intramolecular_forces(particles *p);
double particles_compute_forces_electrostatic_correction_spread(particles *p, grid *g);
double particles_compute_forces_electrostatic_correction_sr(particles *p, grid *g);
double particles_compute_forces_pb(particles *p, grid *grid);
void particles_compute_forces_tot(particles *p);

double particles_get_temperature(particles *p);
double particles_get_kinetic_energy(particles *p);
void particles_get_momentum(particles *p, double *out);
void particles_rescale_velocities(particles *p);
void particles_rescale_momenta(particles *p);
void particles_zero_linear(particles *p);

void ovrvo_integrator_init(integrator *integrator);
void ovrvo_integrator_part1(integrator *integrator, particles *p);
void ovrvo_integrator_part2(integrator *integrator, particles *p);
void ovrvo_integrator_init_thermostat(integrator *integrator, double *params);
void ovrvo_integrator_stop_thermostat(integrator *integrator);

void verlet_integrator_init(integrator *integrator);
void verlet_integrator_part1(integrator *integrator, particles *p);
void verlet_integrator_part2(integrator *integrator, particles *p);
void verlet_integrator_init_thermostat(integrator *integrator, double *params);
void verlet_integrator_stop_thermostat(integrator *integrator);

// Preconditioner function definitions
void precond_jacobi_apply(double *in, double *out, int s1, int s2, int n_start);
void precond_mg_apply(double *in, double *out, int s1, int s2, int n_start);
void precond_ssor_apply(double *in, double *out, int s1, int s2, int n_start);
void precond_blockjacobi_apply(double *in, double *out, int s1, int s2, int n_start);

// Smoothing function definitions
void smooth_charges_none(grid *grid);
// void smooth_charges_gauss(grid *grid, particles *p);
void smooth_charges_diffusion(grid *grid);
void smooth_charges_wendland_nofft_init(grid *grid, int order);
void smooth_charges_wendland_nofft(grid *grid);
void smooth_charges_wendland_nofft_free(grid *grid);

void precond_blockjacobi_init();
void precond_blockjacobi_cleanup();

char *get_water_electrostatic_type_str(int n);

// Struct definitions
struct grid {
    grid_type type;  // Type of the grid
    int n;  // Number of grid points per dimension
    double L;  // Length of the grid
    double h;  // Grid spacing
    double eps_s;  // Dielectric constant of the solvent
    double eps_int;  // Dielectric constant inside the solute

    long int size;  // Total number of grid points
    int n_local; // X - Number of grid points per dimension (MPI aware)
    int n_start; // Start index of the grid in the global array (MPI aware)

    // TODO: Generalize y-history beyond MG so y_hist[0] is the current y,
    // removing the separate y buffer and extrapolating from the first history entries.
    double *y;  // Intermediate field constraint
    // y_hist[0..MAX-1] stores older states, y_hist[MAX] is the scratch/newest slot.
    double *y_hist[MAZE_Y_HIST_MAX + 1];
    int y_hist_len;
    y_extrap_order y_extrap_order;
    double *q;  // Charge density
    double *phi_p;  // Previous potential (could be NULL if not needed by the method)
    double *phi_n;  // Last potential
    // For P3M: phi_p is the newest old field, phi_hist[0..2] are older
    // fields, and phi_hist[3] is a scratch buffer.
    double *phi_hist[MAZE_PHI_HIST_MAX];
    int phi_hist_len;
    int phi_initialized;
    phi_extrap_order phi_extrap_order;
    double *ig2;  // Inverse of the laplacian
    unsigned int *region;  // Region type for each grid point (0=outside, 1=inside) defined in grid nodes

    precond_type precond_type;  // Type of the preconditioner

    // Poisson-Boltzmann specific
    int pb_enabled;  // Poisson-Boltzmann enabled
    int nonpolar_enabled; // Nonpolar forces enabled
    int eps_field_dep_enabled; // Field-dependent dielectric enabled
    int eps_map_type; // Dielectric map construction method
    int pb_force_type; // Poisson-Boltzmann force computation method
    int stress_tensor_bc_type; // Boundary condition used by stress-tensor PB forces
    double w;  // Ionic boundary width
    double kbar2;  // Screening factor
    double kBT;  // Thermal energy factor for field-dependent dielectric updates
    double eps_field_alpha;  // Alpha parameter in eps(E) model (Hu & Wei Eq. S2)
    double *k2;  // Screening factor
    double *eps_x;  // Dielectric constant
    double *eps_y;  // Dielectric constant
    double *eps_z;  // Dielectric constant

    // P3M specific
    int force_gradient_order;
    smoothing_type smoothing;
    void *smoothing_kernel;  // Backend-specific precomputed smoothing data
    double smoothing_rcut;
    double smoothing_sigma;       // Physical screening width; what the short-range correction assumes
    int smoothing_window_order;   // B-spline order P to deconvolve; 0 disables the correction
    double *deconv_scratch;       // Persistent scratch for grid_deconvolve_window (per-step hot path)

    double tol;  // Tolerance for the LCG
    long int n_iters;  // Number of iterations for convergence of the LCG
    int eps_phi_iters;  // Iterations for eps-phi self-consistency (field-dependent dielectric)

    void    (*free)( grid *);
    void    (*init_field)( grid *);
    void    (*apply_precond)( double *, double *, int, int, int);
    int     (*update_field)( grid *);
    double  (*update_charges)( grid *, particles *);
    void    (*update_eps_and_k2)( grid *, particles *);
    void    (*smooth_charges)( grid *);
};

struct neighbor {
    long int idx;  // Index of the neighbor particle
    double dist;  // Distance to the neighbor particle
    double dx;  // X distance to the neighbor particle
    double dy;  // Y distance to the neighbor particle
    double dz;  // Z distance to the neighbor particle

    // Flag to indicate if the neighbor list should stop or continue (for Verlet lists)
    // Needed to be able to re-use the same list while still allowing a proper free at the end
    int valid;
    neighbor *next;  // Pointer to the next neighbor in the list
};

struct particles {
    // int n;  // Number of grid points per dimension
    int n_p;  // Number of particles
    int n_typ;  // Number of particle types (charge, masses, ... definitions)
    double L;  // Length of the grid
    double h;  // Grid spacing

    int np_local; // Number of particles local to the MPI process
    int np_start; // Start index of the particles in the global array (MPI aware)

    int num_neighbors;  // Number of neighbors per particle

    ca_scheme_type cas_type;  // Type of the charge assignment scheme

    int *types;  // Particle types (n_p)
    double *pos;  // Particle positions (n_p x 3)
    double *vel;  // Particle velocities (n_p x 3)
    double *fcs_elec;  // Particle electric forces (n_p x 3)
    double *fcs_noel;  // Particle non-electric forces (n_p x 3)
    double *fcs_tot;  // Particle total forces (n_p x 3)
    double *mass;  // Particle masses (n_p)
    double *charges;  // Particle charges (n_p)
    long int *grid_neighbors;  // Particle neighbors (n_p x 8 x 3)

    int is_water;  // Flag to toggle water/SPC setup
    water_electrostatic_type corr_type; // Type of electrostatic correction for water
    double *fcs_intra; // Intramolecular forces total (n_p x 3)
    double *fcs_corr; // Electrostatic correction forces (n_p x 3)
    double energy_intra; // Intramolecular energy total
    double energy_corr; // Intramolecular exclusion correction energy

    particle_neighbor_type particle_neighbor_method; // Method for finding particle neighbors
    double r_cut;
    neighbor **particle_neighbors;  // Linked list of neighbors for each particle (n_p)
    int cell_list_size;  // Cell size for cell list neighbor finding
    double cell_list_length;  // Cell size for cell list neighbor finding
    long int *cell_list_head;  // Cell list head for cell list neighbor finding (n_cells)
    long int *cell_list_next;  // Cell list next for cell list neighbor finding (n_p)

    potential_type pot_type;  // Type of the potential
    int lj_force_shift;
    double *tf_params;  // Parameters for the TF potential (7 x n_p x n_p)
    double *lj_params;  // Parameters for the LJ potential (4 x n_p x n_p)
    double *sc_params;  // Parameters for the SC potential (5)

    // Poisson-Boltzmann specific
    int pb_enabled;  // Poisson-Boltzmann enabled
    int nonpolar_enabled; // Nonpolar forces enabled
    double gamma_np;
    double beta_np;
    // double *fcs_rf; // Particle reaction field forces (n_p x 3)
    double *fcs_db; // Dielectric boundary forces (n_p x 3)
    double *fcs_ib; // Ionic boundary forces (n_p x 3)
    double *fcs_np; // Non-polar forces (n_p x 3)
    double *solv_radii; // Solvation radii for each particle (n_p)

    void    (*free)( particles *);

    void    (*init_potential)( particles *, int, double *);

    void    (*update_particle_neighbors)( particles *);
    void    (*update_grid_nearest_neighbors)( particles *, grid *);
    double  (*charges_spread_func)( double, double, double);


    double  (*compute_forces_field)( particles *, grid *);
    double  (*compute_forces_noel)( particles *);
    void    (*compute_forces_tot)( particles *);
    double  (*compute_forces_pb)( particles *, grid *);
    double  (*compute_intramolecular_forces)( particles *);
    double  (*compute_forces_electrostatic_correction)( particles *, grid *);

    double  (*get_temperature)( particles *);
    double  (*get_kinetic_energy)( particles *);
    void    (*get_momentum)( particles *, double *);

    void    (*rescale_velocities)( particles *);
    void    (*rescale_momenta)( particles *);
};

struct integrator {
    integrator_type type;  // Type of the integrator
    int n_p;  // Number of particles
    double dt;  // Time step
    double T;  // Temperature

    int enabled;  // Thermostat enabled
    double c1;  // Thermostat parameter
    double c2;  // Thermostat parameter

    void    (*part1)( integrator *, particles *);
    void    (*part2)( integrator *, particles *);
    void    (*init_thermostat)( integrator *, double *);
    void    (*stop_thermostat)( integrator *);
    void    (*free)( integrator *);
};

#endif // __MP_STRUCTS_H
