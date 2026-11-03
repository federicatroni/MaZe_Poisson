#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#include "mpi_base.h"
#include "mp_structs.h"
#include "sphere_intersect.h"

static int pbc_grid_index(int idx, int n) {
    idx %= n;
    if (idx < 0) idx += n;
    return idx;
}

char grid_type_str[GRID_TYPE_NUM][16] = {"LCG", "FFT", "MULTIGRID", "MAZE-LCG", "MAZE-MULTIGRID"}; 
int get_grid_type_num() {
    return GRID_TYPE_NUM;
}
char *get_grid_type_str(int n) {
    return grid_type_str[n];
}

char precond_type_str[PRECOND_TYPE_NUM][16] = {"NONE", "JACOBI", "MG", "SSOR", "BLOCKJACOBI"};
int get_precond_type_num() {
    return PRECOND_TYPE_NUM;
}
char *get_precond_type_str(int n) {
    return precond_type_str[n];
}

char eps_map_type_str[EPS_MAP_TYPE_NUM][32] = {"TRADITIONAL", "SPHERE", "FIELD_DEPENDENT"};
int get_eps_map_type_num() {
    return EPS_MAP_TYPE_NUM;
}
char *get_eps_map_type_str(int n) {
    return eps_map_type_str[n];
}

char pb_force_type_str[PB_FORCE_TYPE_NUM][32] = {"PB_ROUX", "STRESS_TENSOR"};
int get_pb_force_type_num() {
    return PB_FORCE_TYPE_NUM;
}
char *get_pb_force_type_str(int n) {
    return pb_force_type_str[n];
}

char stress_tensor_bc_type_str[STRESS_TENSOR_BC_TYPE_NUM][32] = {"DBC", "PBC"};
int get_stress_tensor_bc_type_num() {
    return STRESS_TENSOR_BC_TYPE_NUM;
}
char *get_stress_tensor_bc_type_str(int n) {
    return stress_tensor_bc_type_str[n];
}

grid * grid_init(int n, double L, double h, double tol, double eps, double eps_int, int grid_type, int precond_type) {
    void   (*init_func)(grid *);
    switch (grid_type) {
        case GRID_TYPE_LCG:
            init_func = lcg_grid_init;
            break;
        case GRID_TYPE_FFT:
            init_func = fft_grid_init;
            break;
        case GRID_TYPE_MGRID:
            init_func = multigrid_grid_init;  // Assuming multigrid_init is defined elsewhere
            break;
        case GRID_TYPE_MAZE_LCG:
            init_func = maze_lcg_grid_init;  
            break;
        case GRID_TYPE_MAZE_MGRID:
            init_func = maze_multigrid_grid_init;  
            break;
        default:
            break;
    }

    grid *new = (grid *)malloc(sizeof(grid));
    new->type = grid_type;
    new->precond_type = precond_type;
    new->n = n;
    new->L = L;
    new->h = h;
    new->eps_s = eps;  // Dielectric constant of the solvent
    {   // [slit_dsl patch] uniaxial solvent: eps_s on x,y edges, MAZE_EPS_PERP on z edges
        const char *ep = getenv("MAZE_EPS_PERP");
        new->eps_perp = (ep != NULL && atof(ep) > 0.0) ? atof(ep) : eps;
    }
    new->eps_int = eps_int;  // Dielectric constant inside the solute

    new->n_local = n;
    new->n_start = 0;

    new->y = NULL;
    new->q = NULL;
    new->phi_p = NULL;
    new->phi_n = NULL;
    new->ig2 = NULL;
    new->region = NULL;
    new->st_owner = NULL;


    new->pb_enabled = 0;  // Poisson-Boltzmann not enabled by default
    new->nonpolar_enabled = 0; //nonpolar forces not enabled by default
    new->eps_field_dep_enabled = 0; //field-dependent dielectric not enabled by default

    // These are set in `grid_pb_init` which should always be called before running PB related functions
    // Using this allows us to give a more descriptive error message
    new->eps_map_type = MAP_NOT_INITIALIZED;
    new->pb_force_type = MAP_NOT_INITIALIZED;
    new->stress_tensor_bc_type = MAP_NOT_INITIALIZED;

    new->w = 0.0;  // Ionic boundary width
    new->kbar2 = 0.0;  // Screening factor
    new->kBT = 0.0;
    new->eps_field_alpha = 1.0;

    new->k2 = NULL;  // Screening factor
    new->eps_x = NULL;  // Dielectric constant in x direction
    new->eps_y = NULL;  // Dielectric constant in y direction
    new->eps_z = NULL;  // Dielectric constant in z direction

    new->update_field = NULL;
    new->update_charges = NULL;
    new->update_eps_and_k2 = NULL;
    
    init_func(new);

    new->tol = tol;
    new->n_iters = 0;
    new->field_update_count = 0;
    new->eps_phi_iters = 0;

    new->free = grid_free;

    return new;
}

void grid_pb_init(
    grid *grid, double w, double kbar2, int nonpolar_enabled, int eps_map_type, int pb_force_type, int stress_tensor_bc_type, double kBT, double eps_field_alpha
) {
    // Initialize the grid for Poisson-Boltzmann simulations
    grid->pb_enabled = 1;  // Enable Poisson-Boltzmann
    grid->nonpolar_enabled = nonpolar_enabled; //nonpolar forces ON/OFF
    grid->eps_map_type = eps_map_type;
    grid->eps_field_dep_enabled = (eps_map_type == EPS_MAP_TYPE_FIELD_DEPENDENT);
    grid->pb_force_type = pb_force_type;
    grid->stress_tensor_bc_type = stress_tensor_bc_type;
    grid->w = w;
    grid->kbar2 = kbar2;
    grid->kBT = kBT;
    grid->eps_field_alpha = eps_field_alpha;

    // Initialize the solvent potential and dielectric constant arrays
    int n = grid->n;
    int n_local = grid->n_local;

    grid->update_eps_and_k2 = grid_update_eps_and_k2;

    grid->eps_x = mpi_grid_allocate(n_local, n);
    grid->eps_y = mpi_grid_allocate(n_local, n);
    grid->eps_z = mpi_grid_allocate(n_local, n);
    grid->k2 = (double *)malloc(grid->size * sizeof(double));
    grid->region = mpi_grid_allocate_uint(n_local, n);
    grid->st_owner = mpi_grid_allocate_uint(n_local, n);
    // Only `grid_update_eps_and_k2_sphere` fills the owner map. Leaving it at
    // ST_OWNER_NONE makes the stress-tensor forces ignore it for the other
    // dielectric maps, which keeps their behaviour unchanged.
    memset(grid->st_owner - (long)n * n, 0xFF,
           (size_t)(n_local + 2) * (size_t)n * (size_t)n * sizeof(unsigned int));
}

void grid_pb_free(grid *grid) {
    if (grid->pb_enabled) {
        mpi_grid_free(grid->eps_x, grid->n);
        mpi_grid_free(grid->eps_y, grid->n);
        mpi_grid_free(grid->eps_z, grid->n);

        free(grid->k2);
        mpi_grid_free_uint(grid->region, grid->n);
        mpi_grid_free_uint(grid->st_owner, grid->n);
    }
}

void grid_free(grid *grid) {
    switch (grid->type) {
        case GRID_TYPE_LCG:
            lcg_grid_cleanup(grid);
            break;
        case GRID_TYPE_FFT:
            fft_grid_cleanup(grid);
            break;
        case GRID_TYPE_MGRID:
            multigrid_grid_cleanup(grid);
            break;
        case GRID_TYPE_MAZE_LCG:
            maze_lcg_grid_cleanup(grid);
            break;
        case GRID_TYPE_MAZE_MGRID:
            maze_multigrid_grid_cleanup(grid);
            break;
        default:
            break;
    }

    grid_pb_free(grid);

    free(grid);
}

void grid_update_eps_and_k2_roux(grid *g, particles *p) {
    // Update the dielectric constant and screening factor based on the grid's transition regions
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

    long int size = g->size;
    double *k2 = g->k2;
    double *eps_x = g->eps_x;
    double *eps_y = g->eps_y;
    double *eps_z = g->eps_z;

    #pragma omp parallel for
    for (long int i = 0; i < size; i++) {
        eps_x[i] = (eps_s - eps_int);
        eps_y[i] = (eps_s - eps_int);
        eps_z[i] = (eps_s - eps_int);
        k2[i] = kbar2;  // Update screening factor
    }

    // #pragma \
    //     omp parallel for private(r_solv, px, py, pz, idx_x, idx_y, idx_z) \
    //     reduction(*:k2[:size], eps_x[:size], eps_y[:size], eps_z[:size])
    for (int np = 0; np < p->n_p; np++) {
        r_solv = p->solv_radii[np];
        px = p->pos[np * 3];
        py = p->pos[np * 3 + 1];
        pz = p->pos[np * 3 + 2];


        double r2;
        double r_solv_p2 = pow(r_solv + w, 2);
        double r_solv_m2 = pow(r_solv - w, 2);

        int idx_range = (int)floor((r_solv + w) / h) + 1;

        idx_x = (int)floor(px / h);
        idx_y = (int)floor(py / h);
        idx_z = (int)floor(pz / h);

        double dx, dy, dz;
        double dx2, dy2, dz2;
        double app1, app2;

        int i0, j0, k0;
        long int idx_cen;
        
        for (int di = -idx_range; di <= idx_range; di++) {
            i0 = idx_x + di;
            dx = px - i0 * h;  // Calculate the distance in x direction
            dx2 = dx * dx;
            i0 = (i0 + n) % n;  // Wrap around for periodic boundary conditions
            i0 -= n_start;  // Adjust for local grid start
            if (i0 < 0 || i0 >= n_local) continue;  // Skip if the point is outside the local grid
            i0 *= n2;  // Convert to linear index
            for (int dj = -idx_range; dj <= idx_range; dj++) {
                j0 = idx_y + dj;
                dy = py - j0 * h;  // Calculate the distance in y direction
                dy2 = dy * dy;
                j0 = (j0 + n) % n;  // Wrap around for periodic boundary conditions
                j0 *= n;
                for (int dk = -idx_range; dk <= idx_range; dk++) {
                    k0 = idx_z + dk;
                    dz = pz - k0 * h;  // Calculate the distance in z direction
                    dz2 = dz * dz;
                    k0 = (k0 + n) % n;  // Wrap around for periodic boundary conditions

                    r2 = dx2 + dy2 + dz2;

                    idx_cen = i0 + j0 + k0;  // Calculate the index in the grid

                    if (r2 >= r_solv_p2) {
                        // Outside the radius, skip this point
                        // continue;  // Skip if outside the radius
                    } else if (r2 > r_solv_m2) {
                        // Inside the transition region, set dielectric constant to a fraction
                        app2 = sqrt(r2) - r_solv + w;  // Calculate the distance in the transition region
                        k2[idx_cen] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        k2[idx_cen] = 0.0;  // Set screening factor to zero
                    }

                    // *************** X + h/2 ***************
                    app1 = dx - hd2;  // Adjust for half the grid spacing
                    r2 = app1 * app1 + dy2 + dz2;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        app2 = sqrt(r2) - r_solv + w;
                        eps_x[idx_cen] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        eps_x[idx_cen] = 0.0;
                    }

                    // *************** Y + h/2 ***************
                    app1 = dy - hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + app1 * app1 + dz2;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        app2 = sqrt(r2) - r_solv + w;
                        eps_y[idx_cen] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        eps_y[idx_cen] = 0.0;
                    }

                    // *************** Z + h/2 ***************
                    app1 = dz - hd2;  // Adjust for half the grid spacing
                    r2 = dx2 + dy2 + app1 * app1;
                    if (r2 >= r_solv_p2) {
                        // Do nothihng
                    } else if (r2 > r_solv_m2) {
                        // Apply the transition region formula
                        app2 = sqrt(r2) - r_solv + w;
                        eps_z[idx_cen] *= (
                            -(1 / (4 * w3)) * pow(app2, 3) +
                             (3 / (4 * w2)) * pow(app2, 2) 
                        );
                    } else {
                        // Inside the radius, set dielectric constant to zero
                        eps_z[idx_cen] = 0.0;
                    }
                }
            }
        }
    }
    for (long int i = 0; i < size; i++) {
        eps_x[i] += eps_int;  // Update x dielectric constant
        eps_y[i] += eps_int;  // Update y dielectric constant
        eps_z[i] += eps_int;  // Update z dielectric constant
    }
}    

double grid_update_eps_field_dependent(grid *g, particles *p) {
    mpi_fprintf(stderr, "This function should not be used YET!!\n");
    exit(1);
    int n = g->n;
    int n_local = g->n_local;
    double h = g->h;
    double kBT = g->kBT;

    double eps_s   = g->eps_s;
    double eps_int = g->eps_int;

    long int n2   = (long int)n * (long int)n;
    long int size = g->size;

    double *eps_x = g->eps_x;
    double *eps_y = g->eps_y;
    double *eps_z = g->eps_z;
    double *phi_n = g->phi_n;
    
    double alpha = g->eps_field_alpha;  /* Hu & Wei Eq. S2 parameter (user-configurable) */
    // Hu & Wei SI (Eq. S2, n=1): eps = eps_m + (eps_s-eps_m)/(1 + (alpha/(2 kBT)) * |grad phi|^2)
    // The previous implementation used 1/(2*alpha*kBT)^2, which is not consistent with Eq. S2.
    double inv_E0 = alpha / (2.0 * kBT);
    double inv_two_h = 1.0 / (2.0 * h);
    double inv_h     = 1.0 / h;
    double delta_eps = eps_s - eps_int;

    double max_diff = 0.0;

    // Ensure halo planes of phi are available for local boundary gradients
    mpi_grid_exchange_bot_top(phi_n, n_local, n);

    #pragma omp parallel for reduction(max:max_diff)
    for (long int idx = 0; idx < size; idx++) {
        long int ix = idx / n2;
        long int iy = (idx % n2) / n;
        long int iz = idx % n;

        // x-direction uses halo planes (no local wrap)
        long int ixp = ix + 1;
        long int ixm = ix - 1;
        if (ixp == n_local) ixp = n_local;  // top halo plane
        if (ixm < 0)        ixm = -1;       // bottom halo plane

        long int iyp = iy + 1; if (iyp == n) iyp = 0;
        long int iym = iy - 1; if (iym < 0)  iym = n - 1;
        long int izp = iz + 1; if (izp == n) izp = 0;
        long int izm = iz - 1; if (izm < 0)  izm = n - 1;

        double phi_ijk     = phi_n[idx];
        double phi_iplus1  = phi_n[ixp * n2 + iy  * n + iz];
        double phi_iminus1 = phi_n[ixm * n2 + iy  * n + iz];
        double phi_jplus1  = phi_n[ix  * n2 + iyp * n + iz];
        double phi_jminus1 = phi_n[ix  * n2 + iym * n + iz];
        double phi_kplus1  = phi_n[ix  * n2 + iy  * n + izp];
        double phi_kminus1 = phi_n[ix  * n2 + iy  * n + izm];

        double Ex_half = (phi_iplus1  - phi_ijk) * inv_h;
        double Ey_half = (phi_jplus1  - phi_ijk) * inv_h;
        double Ez_half = (phi_kplus1  - phi_ijk) * inv_h;

        double Ex = (phi_iplus1  - phi_iminus1) * inv_two_h;
        double Ey = (phi_jplus1  - phi_jminus1) * inv_two_h;
        double Ez = (phi_kplus1  - phi_kminus1) * inv_two_h;

        double E_mag_x2 = Ex_half * Ex_half + Ey * Ey + Ez * Ez;
        double E_mag_y2 = Ex * Ex + Ey_half * Ey_half + Ez * Ez;
        double E_mag_z2 = Ex * Ex + Ey * Ey + Ez_half * Ez_half;
        
        double new_x = eps_int + delta_eps / (1.0 + E_mag_x2 * inv_E0);
        double new_y = eps_int + delta_eps / (1.0 + E_mag_y2 * inv_E0);
        double new_z = eps_int + delta_eps / (1.0 + E_mag_z2 * inv_E0);
        // printf("index %ld: Ex^2=%e, (Ex/E0)^2=%e,  eps(E)=%lf\n", idx, E_mag_x2, E_mag_x2 * inv_E02, new_x);

        double dx = fabs(new_x - eps_x[idx]);
        double dy = fabs(new_y - eps_y[idx]);
        double dz = fabs(new_z - eps_z[idx]);

        double local_max = dx;
        if (dy > local_max) local_max = dy;
        if (dz > local_max) local_max = dz;

        if (local_max > max_diff){
            max_diff = local_max;
            // mpi_printf("Max dielectric change updated: %e at index %ld\n", max_diff, idx);
        }
        eps_x[idx] = new_x;
        eps_y[idx] = new_y;
        eps_z[idx] = new_z;
    }
    allreduce_max(&max_diff, 1);
    // mpi_printf("\nMaximum dielectric constant change after update: %e\n", max_diff);
    return max_diff;
}

double wha (double eps1, double eps2, double frac)
{
    return 1.0 / (frac / eps1 + (1.0 - frac) / eps2);
}

static double eps_mix_eval(double eps1, double eps2, double frac)
{
    return wha(eps1, eps2, frac);
}

/* [slit_dsl patch 2026-10-14] dielectric slab: env MAZE_EPS_OUT = eps_o switches on a different isotropic solvent
 * outside the slit |z - zc| > H (zc, H from MAZE_WALL, Angstrom). Inside: uniaxial (eps_s on x,y edges, eps_perp on
 * z edges). x,y edges lying exactly on the interface plane get the arithmetic mean (parallel capacitors); z edges
 * crossing it the harmonic mean weighted by the inside fraction (series). Unset -> uniform medium, original behaviour.
 * slab_eps_layers() fills, per grid layer k, the solvent eps of the x,y edges (sx) and of the z edge k -> k+1 (sz). */
static int slab_eps_init = 0, slab_eps_on = 0;
static double slab_zc = 0.0, slab_H = 0.0, slab_eps_o = 1.0;
static void slab_eps_setup(void) {
    slab_eps_init = 1;
    const char *e = getenv("MAZE_EPS_OUT"), *w = getenv("MAZE_WALL");
    if (e == NULL) return;
    if (w == NULL) { mpi_fprintf(stderr, "MAZE_EPS_OUT needs MAZE_WALL (zc,H)\n"); exit(1); }
    const double bohr_A = 0.529177210903;
    double zc_A = 0.0, H_A = 0.0;
    if (sscanf(w, "%lf,%lf", &zc_A, &H_A) != 2) { mpi_fprintf(stderr, "MAZE_WALL: cannot read zc,H\n"); exit(1); }
    const char *he = getenv("MAZE_EPS_H");   // [slit_dsl patch 2026-10-15] dielectric interface at zc +- MAZE_EPS_H (A), walls stay at MAZE_WALL H
    if (he != NULL) H_A = atof(he);
    slab_zc = zc_A / bohr_A; slab_H = H_A / bohr_A; slab_eps_o = atof(e); slab_eps_on = 1;
    mpi_printf("[slit_dsl] dielectric slab: inside |z-%g|<%g A uniaxial, outside eps_o=%g\n", zc_A, H_A, slab_eps_o);
}
int slab_eps_active(double *zc, double *H, double *eps_o) {
    if (!slab_eps_init) slab_eps_setup();
    if (slab_eps_on) { *zc = slab_zc; *H = slab_H; *eps_o = slab_eps_o; }
    return slab_eps_on;
}
/* [electrode patch 2026-10-17] Thomas-Fermi electrodes at fixed potential. Env MAZE_ELECTRODE = "t,lTF,psi_top,psi_bot"
 * (A, A, V, V; needs MAZE_EPS_OUT). Electrode nodes: H <= |z - zc| < H + t (zc, H of the dielectric slab), top for z > zc.
 * There eps = eps_o and k2 = eps_o h^2 / lTF^2 (array units: k2 = kappa^2 h^2), i.e. div(eps grad phi) - eps_o/lTF^2 (phi - Psi) = -4 pi rho.
 * Psi enters the RHS as a fixed grid charge q_e = k2 Psi h / (4 pi) added to grid->q after the ion charges (A phi + 4 pi q/h = 0).
 * The rest of the cell (|z - zc| >= H + t) is an eps_o spacer that insulates the two electrodes across the periodic boundary.
 * Integrating the equation over the cell gives total induced charge = -Q_ions (global electroneutrality, as in MetalWalls).
 * Induced charge per node: -k2 (phi - Psi) h / (4 pi). Recorded per electrode and face (inner half / outer half of the layer)
 * every MAZE_ELEC_STRIDE field updates to MAZE_ELEC_OUT (default electrode.csv): call,Qtop_in,Qtop_out,Qbot_in,Qbot_out [e]. */
static int el_init = 0, el_on = 0, el_stride = 10;
static double el_t = 0.0, el_k2 = 0.0, el_psi[2] = {0.0, 0.0};
static long el_calls = 0;
static FILE *el_fp = NULL;
static void electrode_setup(void) {
    el_init = 1;
    const char *e = getenv("MAZE_ELECTRODE");
    if (e == NULL) return;
    const double bohr_A = 0.529177210903, Ha_V = 27.211386245988;
    double t, l, pt, pb;
    if (sscanf(e, "%lf,%lf,%lf,%lf", &t, &l, &pt, &pb) != 4) { mpi_fprintf(stderr, "MAZE_ELECTRODE: need t,lTF,psi_top,psi_bot\n"); exit(1); }
    double zc, H, eo;
    if (!slab_eps_active(&zc, &H, &eo)) { mpi_fprintf(stderr, "MAZE_ELECTRODE needs MAZE_EPS_OUT and MAZE_WALL\n"); exit(1); }
    if (getenv("MAZE_SLAB_CORR") != NULL) { mpi_fprintf(stderr, "MAZE_ELECTRODE: unset MAZE_SLAB_CORR (electrodes fix the k=0 mode)\n"); exit(1); }
    el_t = t / bohr_A; el_k2 = eo / ((l / bohr_A) * (l / bohr_A));   /* times h^2 when applied */
    el_psi[0] = pt / Ha_V; el_psi[1] = pb / Ha_V;
    const char *s = getenv("MAZE_ELEC_STRIDE"); if (s != NULL && atoi(s) > 0) el_stride = atoi(s);
    const char *o = getenv("MAZE_ELEC_OUT");
    el_fp = fopen(o != NULL ? o : "electrode.csv", "w");
    if (el_fp == NULL) { mpi_fprintf(stderr, "MAZE_ELECTRODE: cannot open output file\n"); exit(1); }
    fprintf(el_fp, "call,Qtop_in,Qtop_out,Qbot_in,Qbot_out\n");
    el_on = 1;
    mpi_printf("[electrode] TF electrodes |z-zc| in [%g, %g) A, lTF %g A, eps %g, Psi top %g V bottom %g V\n", H * bohr_A, H * bohr_A + t, l, eo, pt, pb);
}
int electrode_active(void) { if (!el_init) electrode_setup(); return el_on; }
/* [st wall patch, metal] induced charge of a node of plane k (e), 0 outside the electrodes */
static int electrode_class(int k, double h, double L);
double electrode_node_charge(double phi, int k, double h, double L) {
    if (!electrode_active()) return 0.0;
    const int c = electrode_class(k, h, L);
    if (c < 0) return 0.0;
    return -el_k2 * h * h * (phi - el_psi[c / 2]) * h / (4.0 * M_PI);
}
/* layer class of node plane k: 0/1 = top inner/outer, 2/3 = bottom inner/outer, -1 = not an electrode */
static int electrode_class(int k, double h, double L) {
    double z = k * h - slab_zc; z -= L * round(z / L);
    const double a = fabs(z) - slab_H;
    if (a < -1e-9 * h || a >= el_t - 1e-9 * h) return -1;
    return (z > 0 ? 0 : 2) + (a >= 0.5 * el_t ? 1 : 0);
}
void electrode_apply_k2(grid *g) {
    if (!electrode_active()) return;
    const int n = g->n; const double h = g->h, L = g->L;
    for (long idx = 0; idx < g->size; idx++)
        if (electrode_class((int)(idx % n), h, L) >= 0) g->k2[idx] = el_k2 * h * h;
}
void electrode_add_q(grid *g) {
    if (!electrode_active()) return;
    const int n = g->n; const double h = g->h, L = g->L;
    for (long idx = 0; idx < g->size; idx++) {
        const int c = electrode_class((int)(idx % n), h, L);
        if (c >= 0) g->q[idx] += el_k2 * h * h * el_psi[c / 2] * h / (4.0 * M_PI);
    }
}
void electrode_charges(grid *g, double *Q) {   /* Q[4]: top in, top out, bottom in, bottom out (e) */
    for (int c = 0; c < 4; c++) Q[c] = 0.0;
    if (!electrode_active()) return;
    const int n = g->n; const double h = g->h, L = g->L, k2 = el_k2 * h * h;
    for (long idx = 0; idx < g->size; idx++) {
        const int c = electrode_class((int)(idx % n), h, L);
        if (c >= 0) Q[c] -= k2 * (g->phi_n[idx] - el_psi[c / 2]) * h / (4.0 * M_PI);
    }
    allreduce_sum(Q, 4);
}
void electrode_record(grid *g) {
    if (!electrode_active()) return;
    double Q[4];
    if (el_calls % el_stride == 0) {
        electrode_charges(g, Q);
        fprintf(el_fp, "%ld,%.8f,%.8f,%.8f,%.8f\n", el_calls, Q[0], Q[1], Q[2], Q[3]);
        fflush(el_fp);
    }
    el_calls++;
}
/* [st wall patch 2026-10-31] MAZE_ST_WALL=1 (dielectric slab, sharp interface, no electrodes): the stress-tensor control volume of
 * an ion that reaches the interface is closed INSIDE the barrier (first node plane beyond the interface, same columns as the
 * owned interface-plane nodes) and the force of the enclosed interface patch is subtracted in forces.c (validated in Python:
 * notes/application/slit_dsl/wall_st, error <= 1% vs FD down to contact). The ST then needs the background (cavity-free)
 * eps of every edge: slab_bg_layers() returns the last per-layer arrays. */
static double *bg_sx = NULL, *bg_sz = NULL; static int bg_n = 0;
int st_wall_active(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("MAZE_ST_WALL"); on = (e != NULL && atoi(e) != 0); }
    return on;
}
int slab_bg_layers(const double **sx, const double **sz, double *zc, double *H, double *eps_o) {
    if (bg_sx == NULL || !slab_eps_active(zc, H, eps_o)) return 0;
    *sx = bg_sx; *sz = bg_sz; return bg_n;
}
static void slab_eps_layers_impl(int n, double h, double L, double eps_s, double eps_perp, double *sx, double *sz);
static void slab_eps_layers(int n, double h, double L, double eps_s, double eps_perp, double *sx, double *sz) {
    slab_eps_layers_impl(n, h, L, eps_s, eps_perp, sx, sz);
    if (bg_n != n) { free(bg_sx); free(bg_sz); bg_sx = malloc(n * sizeof(double)); bg_sz = malloc(n * sizeof(double)); bg_n = n; }
    memcpy(bg_sx, sx, n * sizeof(double)); memcpy(bg_sz, sz, n * sizeof(double));
}
static void slab_eps_layers_impl(int n, double h, double L, double eps_s, double eps_perp, double *sx, double *sz) {
    double zc, H, eo;
    if (!slab_eps_active(&zc, &H, &eo)) {
        for (int k = 0; k < n; k++) { sx[k] = eps_s; sz[k] = eps_perp; }
        return;
    }
    const double tol = 1e-6 * h;
    for (int k = 0; k < n; k++) {
        double d = k * h - zc; d -= L * round(d / L);
        double ad = fabs(d);
        sx[k] = (ad < H - tol) ? eps_s : (ad > H + tol ? eo : 0.5 * (eps_s + eo));
        double lo = d, hi = d + h;                       // z edge k -> k+1, relative to zc
        double in = fmin(hi, H) - fmax(lo, -H); if (in < 0.0) in = 0.0;
        double f = in / h; if (f > 1.0) f = 1.0;
        sz[k] = 1.0 / (f / eps_perp + (1.0 - f) / eo);
    }
}

void grid_update_eps_and_k2_sphere(grid *g, particles *p)
{
    const int n       = g->n;
    const int n_local = g->n_local;
    const int n_start = g->n_start;
    const double h    = g->h;
    const long size   = g->size;
    const double L    = g->L;

    const double eps_s = g->eps_s;
    const double eps_perp = g->eps_perp;  // [slit_dsl patch] uniaxial background on z edges
    const double eps_m = g->eps_int;
    const double kbar2 = g->kbar2;

    double       *eps_x  = g->eps_x;
    double       *eps_y  = g->eps_y;
    double       *eps_z  = g->eps_z;
    double       *k2     = g->k2;
    unsigned int *region = g->region;
    unsigned int *owner  = g->st_owner;
    double *slx = (double *)malloc((size_t)n * sizeof(double)), *slz = (double *)malloc((size_t)n * sizeof(double));
    slab_eps_layers(n, h, L, eps_s, eps_perp, slx, slz);   // [slit_dsl patch] per-layer solvent eps

    /*
     * STEP 2 below always evaluates the analytic edge eps fraction
     * (sphere_edge_fraction) instead of only when the two endpoints of an
     * edge have different region classification. That endpoint-based
     * topology switch is a validated bug, not a design choice: it silently
     * skips chords whose two endpoints are both "outside" but which still
     * cross the sphere (the common case when h is comparable to the gap
     * between two surfaces), producing a force discontinuity whenever a
     * configuration crosses that classification boundary. The analytic fix
     * was measured cost-free and force-identical to the old behaviour
     * everywhere except those mishandled chords. Set MAZE_EPS_LEGACY_EDGES=1
     * to opt back into the old buggy behaviour for regression comparisons.
     */
    const char *legacy_edges_env = getenv("MAZE_EPS_LEGACY_EDGES");
    const int continuous_geometry =
        !(legacy_edges_env != NULL && atoi(legacy_edges_env) != 0);

    /* ====================================================
     * STEP 1 - classify inside/outside using VdW spheres
     * ==================================================== */

    int *line_off = NULL, *line_cand = NULL;
    const int have_lines = continuous_geometry &&
        sphere_build_line_candidates(p, n, n_local, n_start, h, L, &line_off, &line_cand) == 0;

    if (have_lines) {
        /* Fused STEP 1 + STEP 2 on grid lines: same values, only the spheres that can reach a
         * line are evaluated (linear in grid points and particles) */
        const long n_lines = (long)n_local * n;

        #pragma omp parallel for schedule(static)
        for (long line = 0; line < n_lines; line++) {
            const int i = (int)(line / n);
            const int j = (int)(line % n);
            const double x = (i + n_start) * h;
            const double y = j * h;
            const long base = line * n;
            double fx[n], fy[n], fz[n];

            sphere_line_eval_cand(p, x, y, h, L, n,
                line_cand + line_off[line], line_off[line + 1] - line_off[line],
                region + base, fx, fy, fz);

            for (int k = 0; k < n; k++) {
                const long idx = base + k;
                k2[idx] = region[idx] ? 0.0 : kbar2;
                eps_x[idx] = eps_mix_eval(eps_m, slx[k], fx[k]);
                eps_y[idx] = eps_mix_eval(eps_m, slx[k], fy[k]);
                eps_z[idx] = eps_mix_eval(eps_m, slz[k], fz[k]);
            }
        }

        mpi_grid_exchange_bot_top_uint(region, n_local, n);
    } else {
    long long region_inside  = 0;
    long long region_outside = 0;

    #pragma omp parallel for schedule(static) reduction(+:region_inside,region_outside)
    for (long idx = 0; idx < size; idx++) {
        int k = idx % n;
        int j = (idx / n) % n;
        int i = idx / (n * n);

        double x = (i + n_start) * h;
        double y = j * h;
        double z = k * h;

        int inside = is_in_molecule_sphere(p, x, y, z, L);

        k2[idx]     = inside ? 0.0 : kbar2;
        region[idx] = inside ? 1u  : 0u;

        if (inside) region_inside++;
        else        region_outside++;
    }

    // printf("REGION_DEBUG (sphere): inside=%lld outside=%lld (tot=%ld)\n",
        //    region_inside, region_outside, size);

    // Exchange the top and bottom region slices
    mpi_grid_exchange_bot_top_uint(region, n_local, n);

    /* ====================================================
     * STEP 2 - compute epsilon on each edge
     * ==================================================== */

    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < size; idx++) {
        int k = idx % n;
        int j = (idx / n) % n;
        int i = idx / (n * n);

        double x1 = (i + n_start) * h;
        double y1 = j * h;
        double z1 = k * h;

        eps_x[idx] = slx[k];
        eps_y[idx] = slx[k];
        eps_z[idx] = slz[k];

        if (continuous_geometry) {
            double fr[3];
            sphere_edge_fractions3(p, x1, y1, z1, h, L, fr);
            eps_x[idx] = eps_mix_eval(eps_m, slx[k], fr[0]);
            eps_y[idx] = eps_mix_eval(eps_m, slx[k], fr[1]);
            eps_z[idx] = eps_mix_eval(eps_m, slz[k], fr[2]);
            continue;
        }

        /* Edge X */
        long idx_px = idx + (long)n * n;
        if (region[idx] != region[idx_px]) {
            double frac = sphere_edge_fraction(p, x1, y1, z1, h, 0, L);
            eps_x[idx] = eps_mix_eval(eps_m, slx[k], frac);
        } else if (region[idx] != 0) {
            eps_x[idx] = eps_m;
        }

        /* Edge Y */
        long idx_py = (j == n - 1) ? idx - (long)(n - 1) * n : idx + n;
        if (region[idx] != region[idx_py]) {
            double frac = sphere_edge_fraction(p, x1, y1, z1, h, 1, L);
            eps_y[idx] = eps_mix_eval(eps_m, slx[k], frac);
        } else if (region[idx] != 0) {
            eps_y[idx] = eps_m;
        }

        /* Edge Z */
        long idx_pz = (k == n - 1) ? idx - (n - 1) : idx + 1;
        if (region[idx] != region[idx_pz]) {
            double frac = sphere_edge_fraction(p, x1, y1, z1, h, 2, L);
            eps_z[idx] = eps_mix_eval(eps_m, slz[k], frac);
        } else if (region[idx] != 0) {
            eps_z[idx] = eps_m;
        }
    }
    }

    free(line_off);
    free(line_cand);

    /* [slit_dsl patch] MAZE_NO_CAVITY=1: point ions in the (uniaxial) solvent. The dielectric map is
     * uniform (no cavities) but the radii still define the integration balls / owner map of the stress
     * tensor below, so the ST surface stays ~R away from the CIC charge (a 2-cell ball around a point
     * charge gives 15-40% force errors at h = 0.5, checked in Python 2026-10-11). */
    {
        const char *nc = getenv("MAZE_NO_CAVITY");
        if (nc != NULL && atoi(nc) != 0) {
            #pragma omp parallel for schedule(static)
            for (long idx = 0; idx < size; idx++) {
                const int kk = idx % n;
                eps_x[idx] = slx[kk]; eps_y[idx] = slx[kk]; eps_z[idx] = slz[kk];
                region[idx] = 0u; k2[idx] = kbar2;
            }
            mpi_grid_exchange_bot_top_uint(region, n_local, n);
        }
    }

    /* ====================================================
     * STEP 3 — mark enlarged sphere (region = 2) and assign an owner
    Solvent points (region = 0) inside the integration sphere of
    any particle are marked as 2.
    The radius used is the same as in compute_stress_tensor_forces_spherical:
    R = ceil(solv_radii / h) + 2 (in cell units)

    Every node covered by at least one integration sphere is also assigned a
    single owner: the particle whose physical surface is closest to the node,
    ties broken by the lowest index. Using distance-to-surface rather than
    distance-to-centre is essential for particles with unequal radii: it keeps
    the separating wall in the solvent gap instead of pushing it into the
    larger particle's low-dielectric cavity. The owner map defines the control volume
    V_i = {nodes with st_owner == i} that `compute_stress_tensor_forces_pbc`
    integrates the Maxwell stress tensor on: it is the Voronoi cell of i
    restricted to the union of the integration spheres, so its boundary is
    closed however much the spheres interpenetrate, and the wall separating two
    overlapping spheres is a face of both control volumes with opposite normals.
    The comparison uses the true node-centre distance, not the integer offsets,
    so that ties stay rare and do not depend on the lattice orientation.
     * ==================================================== */
    double slab_zc0 = 0.0, slab_H0 = 0.0, slab_eo0 = 1.0;
    const int slab_on = slab_eps_active(&slab_zc0, &slab_H0, &slab_eo0);
    const int st_wall_on = st_wall_active();   /* [st wall patch] dielectric: + first barrier plane below; metal: interface (= first metal) plane only */
    double *owner_metric = (double *)malloc((size_t)size * sizeof(double));
    if (owner_metric == NULL) {
        mpi_fprintf(stderr, "Error: Unable to allocate memory for the integration sphere owner map\n");
        exit(EXIT_FAILURE);
    }
    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < size; idx++) {
        owner[idx] = ST_OWNER_NONE;
        owner_metric[idx] = 0.0;
    }

    for (int q = 0; q < p->n_p; q++) {
        int iq_g = (int)round(p->pos[q * 3 + 0] / h);
        int jq   = (int)round(p->pos[q * 3 + 1] / h);
        int kq   = (int)round(p->pos[q * 3 + 2] / h);
        int Rq   = (int)ceil(p->solv_radii[q] / h) + 2;
        int R2q  = Rq * Rq;

        for (int di = -Rq; di <= Rq; di++) {
            for (int dj = -Rq; dj <= Rq; dj++) {
                for (int dk = -Rq; dk <= Rq; dk++) {
                    if (di * di + dj * dj + dk * dk >= R2q) continue;
                    int ii_g = pbc_grid_index(iq_g + di, n);
                    int jj   = pbc_grid_index(jq + dj, n);
                    int kk   = pbc_grid_index(kq + dk, n);
                    int ii   = ii_g - n_start;
                    if (ii < 0 || ii >= n_local) continue;
                    long idx = (long)kk + (long)jj * n + (long)ii * (long)n * n;
                    if (region[idx] == 0) region[idx] = 2u;
                    if (slab_on) {   // [slit_dsl patch] control volumes stop below the dielectric interface (closed surface in the slab medium)
                        double dzs = kk * h - slab_zc0; dzs -= L * round(dzs / L);
                        if (st_wall_on) { if (fabs(dzs) > slab_H0 + 1e-6 * h) continue; }   // [st wall patch] interface plane allowed
                        else if (fabs(dzs) > slab_H0 - 1e-6 * h) continue;
                    }

                    // Minimum-image distance between the node and the particle
                    double dx = ii_g * h - p->pos[q * 3 + 0];
                    double dy = jj   * h - p->pos[q * 3 + 1];
                    double dz = kk   * h - p->pos[q * 3 + 2];
                    dx -= L * round(dx / L);
                    dy -= L * round(dy / L);
                    dz -= L * round(dz / L);
                    double d2 = dx * dx + dy * dy + dz * dz;
                    double surface_distance = sqrt(d2) - p->solv_radii[q];

                    if (owner[idx] == ST_OWNER_NONE || surface_distance < owner_metric[idx]) {
                        owner[idx] = (unsigned int)q;
                        owner_metric[idx] = surface_distance;
                    }
                }
            }
        }
    }
    free(owner_metric);
    if (st_wall_on && !electrode_active() && slab_on) {
        /* [st wall patch] first barrier plane: owned by the owner of the interface-plane node of the same column, so the
         * enclosed interface patch is exactly the owned interface cross-section and the bottom (top) face lies at z_if -+ 1.5 h */
        for (int kk = 0; kk < n; kk++) {
            double dzs = kk * h - slab_zc0; dzs -= L * round(dzs / L);
            if (fabs(fabs(dzs) - slab_H0) > 1e-6 * h) continue;          // interface plane
            const int ko = pbc_grid_index(kk + (dzs > 0 ? 1 : -1), n);  // first plane in the barrier
            for (int ii = 0; ii < n_local; ii++) for (int jj = 0; jj < n; jj++) {
                const long ia = (long)kk + (long)jj * n + (long)ii * (long)n * n, ib = (long)ko + (long)jj * n + (long)ii * (long)n * n;
                owner[ib] = owner[ia];
            }
        }
    }
    free(slx); free(slz);

    // Exchange the final region map used by the stress tensor
    mpi_grid_exchange_bot_top_uint(region, n_local, n);
    // The owner of the neighbouring node is read too, and it can fall in the
    // ghost slice of the adjacent rank. The reset loop above only covers
    // [0, size), so the ghosts would otherwise stay stale from the second call on.
    mpi_grid_exchange_bot_top_uint(owner, n_local, n);
    electrode_apply_k2(g);   /* [electrode patch] */
}

void grid_update_eps_and_k2(grid *g, particles *p) {
    switch (g->eps_map_type) {
        case MAP_NOT_INITIALIZED:
            mpi_fprintf(stderr, "Error: Epsilon map type not initialized. Please call `grid_pb_init` before running Poisson-Boltzmann related functions.\n");
            exit(1);
        case EPS_MAP_TYPE_TRADITIONAL:
            grid_update_eps_and_k2_roux(g, p);
            break;
        case EPS_MAP_TYPE_FIELD_DEPENDENT:
            // TODO: should this be `grid_update_eps_field_dependent` instead?
            grid_update_eps_and_k2_roux(g, p);
            break;
        case EPS_MAP_TYPE_SPHERE:
            grid_update_eps_and_k2_sphere(g, p);
            break;
        default:
            mpi_fprintf(stderr, "Unknown epsilon map type: %d\n", g->eps_map_type);
            exit(1);
    }
}


/*Important, when called for IO must be called by all procs*/
double grid_get_energy_elec(grid *g){
    double energy = 0.0;

    #pragma omp parallel for reduction(+:energy)
    for (long int i = 0; i < g->size; i++) {
        // Calculate the change in energy due to the Poisson-Boltzmann potential
        energy += 0.5 * g->q[i] * g->phi_n[i];
    }

    allreduce_sum(&energy, 1);

    return energy;
}
