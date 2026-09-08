#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include "mpi_base.h"
#include "linalg.h"
#include "mp_structs.h"
#include "fftw_wrap.h"
#include "sphere_intersect.h"
#include "laplace.h"
#include "smoothing_wendland_poly.h"

static int pbc_grid_index(int idx, int n) {
    idx %= n;
    if (idx < 0) idx += n;
    return idx;
}

static int validate_y_initial_guess(int y_initial_guess) {
    if (y_initial_guess < 0 || y_initial_guess > MAZE_Y_GUESS_MAX) {
        mpi_fprintf(
            stderr,
            "Invalid y_initial_guess=%d (expected 0..%d). Using BASE.\n",
            y_initial_guess, MAZE_Y_GUESS_MAX
        );
        return 0;
    }
    return y_initial_guess;
}

static int validate_phi_initial_guess(int phi_initial_guess) {
    if (phi_initial_guess < 0 || phi_initial_guess > MAZE_PHI_HIST_MAX) {
        mpi_fprintf(
            stderr,
            "Invalid phi_initial_guess=%d (expected 0..%d). Using VERLET.\n",
            phi_initial_guess, MAZE_PHI_HIST_MAX
        );
        return 1;
    }
    return phi_initial_guess;
}


// TODO: Move this stuff to use the nue enums.h format
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

grid * grid_init(
    int n, double L, double h, double tol, double eps, double eps_int,
    grid_type grid_type, precond_type precond_type, int y_initial_guess, int phi_initial_guess,
    electrostatic_discretization_type discretization, int force_gradient_order
) {
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
    new->force_gradient_order = force_gradient_order;
    laplace_set_discretization(discretization);
    new->n = n;
    new->L = L;
    new->h = h;
    new->eps_s = eps;  // Dielectric constant of the solvent
    new->eps_int = eps_int;  // Dielectric constant inside the solute

    new->n_local = n;
    new->n_start = 0;

    new->y = NULL;
    for (int yh = 0; yh <= MAZE_Y_HIST_MAX; yh++) {
        new->y_hist[yh] = NULL;
    }
    new->y_hist_len = 0;
    new->y_extrap_order = validate_y_initial_guess(y_initial_guess);
    new->q = NULL;
    new->phi_p = NULL;
    new->phi_n = NULL;
    for (int ph = 0; ph < MAZE_PHI_HIST_MAX; ph++) {
        new->phi_hist[ph] = NULL;
    }
    new->phi_hist_len = 0;
    new->phi_initialized = 0;
    new->phi_extrap_order = validate_phi_initial_guess(phi_initial_guess);
    new->ig2 = NULL;
    new->region = NULL;


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

    // grid_smoothing_free runs even if smoothing initialization is skipped.
    new->smoothing_window_order = 0;
    new->deconv_scratch = NULL;
    new->smoothing_kernel = NULL;
    
    init_func(new);

    new->tol = tol;
    new->n_iters = 0;
    new->eps_phi_iters = 0;

    new->free = grid_free;

    return new;
}

#ifdef __MPI

void grid_init_mpi(grid *grid) {
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

void grid_init_mpi_fft(grid *grid) {
    mpi_data *mpid = get_mpi_data();
    
    init_rfft(grid->n, &grid->n_local, &grid->n_start);

    int rank = mpid->rank;
    int size = mpid->size;
    int n_loc, n_start;

    mpid->n_loc = grid->n_local;
    mpid->n_start = grid->n_start;
    for (int i=0; i<size; i++) {
        n_loc = mpid->n_loc;
        n_start = mpid->n_start;
        MPI_Bcast(&n_loc, 1, MPI_INT, i, MPI_COMM_WORLD);
        MPI_Bcast(&n_start, 1, MPI_INT, i, MPI_COMM_WORLD);
        mpid->n_loc_list[i] = n_loc;
        mpid->n_start_list[i] = n_start;
        // printf("FFT MPI(%d %d): n_local = %d, n_start = %d\n", rank, i, n_loc, n_start);
    }
    // Check that if some processors have no local grid points they should be skipped
    // from the loop communication
    if (rank < size-1) {
        if (mpid->n_loc_list[rank+1] == 0) {
            mpid->next_rank = 0;
        } 
    }
    if (rank == 0) {
        if (mpid->n_loc_list[size-1] == 0) {
            for (int i=size-1; i>=0; i--) {
                if (mpid->n_loc_list[i] > 0) {
                    mpid->prev_rank = i;
                    break;
                }
            }
        }
    }
}
#else  // __MPI

void grid_init_mpi(grid *grid) {
    mpi_data *mpid = get_mpi_data();
    mpid->n_loc = grid->n;
    mpid->n_start = 0;
}

void grid_init_mpi_fft(grid *grid) {
    mpi_data *mpid = get_mpi_data();

    init_rfft(grid->n, &grid->n_local, &grid->n_start);

    mpid->n_loc = grid->n;
    mpid->n_start = 0;
}

#endif  // __MPI

void grid_pb_init(
    grid *grid, double w, double kbar2, int nonpolar_enabled,
    int eps_map_type, int pb_force_type, int stress_tensor_bc_type, double kBT, double eps_field_alpha
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
}

void grid_pb_free(grid *grid) {
    if (grid->pb_enabled) {
        mpi_grid_free(grid->eps_x, grid->n);
        mpi_grid_free(grid->eps_y, grid->n);
        mpi_grid_free(grid->eps_z, grid->n);

        free(grid->k2);
        mpi_grid_free_uint(grid->region, grid->n);
    }
}

void smooth_charges_none(grid *grid) {
    // No smoothing, return the original charges
}

/*
Allocate and initialize the Gaussian smoothing kernel in Fourier space.
The kernel is generated in real space as a 3D Gaussian function, normalized, and then transformed
to Fourier space using the forward FFT.
The resulting Fourier-space kernel is stored in the grid structure for later use in smoothing the charge distribution.
*/
void smooth_charges_gauss_init(grid *grid) {
    int n = grid->n;
    int nh = n / 2 + 1;
    long int n2 = n * n;
    long int c_size = grid->n_local * nh * n;  // Size of the complex-space grid for the local portion
    long int r_size = grid->n_local * n2;  // Size of the real-space grid for the local portion

    double sigma = grid->smoothing_sigma / grid->h;  // Convert sigma to grid units
    double sigma2 = sigma * sigma;

    double *gaussian_kernel = (double *)calloc(r_size, sizeof(double));

    // Generate the Gaussian kernel in Real space
    int i, j, k;
    int di, dj, dk;
    double ri, rj, r2;
    for (int i_loc = 0; i_loc < grid->n_local; i_loc++) {
        i = grid->n_start + i_loc;
        di = i > n / 2 ? i - n : i;  // Wrap around for periodicity
        ri = di * di;
        for (j = 0; j < n; j++) {
            dj = j > n / 2 ? j - n : j;  // Wrap around for periodicity
            rj = ri + dj * dj;
            for (k = 0; k < n; k++) {
                dk = k > n / 2 ? k - n : k;  // Wrap around for periodicity
                r2 = rj + dk * dk;
                gaussian_kernel[i_loc * n2 + j * n + k] = exp(-(double)r2 / (2 * sigma2));
            }
        }
    }

    // Normalize the kernel
    double sum = 0.0;
    for (i = 0; i < r_size; i++) {
        sum += gaussian_kernel[i];
    }
    allreduce_sum(&sum, 1);
    dscal(gaussian_kernel, 1.0 / sum, r_size);  // Normalize so that the sum of the kernel is 1
    dscal(gaussian_kernel, 1.0 / pow(n, 3), r_size);  // FFT normalization factor

    // Convert the kernel in Fourier space
    grid->smoothing_kernel = malloc(c_size * sizeof(fftw_complex));
    rfft_3d(n, grid->n_local, gaussian_kernel, (fftw_complex *)grid->smoothing_kernel);

    free(gaussian_kernel);
}

/*
Apply the smoothing kernel currently stored in grid->smoothing_kernel (in Fourier space) to the
charge distribution via convolution (element-wise multiplication in Fourier space). This routine
is agnostic to the actual kernel shape, so it is shared by every smoothing method that works by
convolving the charges with a Fourier-space kernel (Gaussian, Wendland C2, ...).
*/
void smooth_charges_fourier_kernel(grid *grid) {
    int n = grid->n;
    int nh = n / 2 + 1;
    int n_loc = grid->n_local;
    int n_start = grid->n_start;
    long int n2 = n * n;
    long int c_size = n_loc * n * nh;  // Size of the complex-space grid for the local portion

    double *q = grid->q;  // Original charge distribution

    // Perform forward FFT on the original charge distribution
    fftw_complex *q_fft = (fftw_complex *)malloc(c_size * sizeof(fftw_complex));
    rfft_3d(n, n_loc, q, q_fft);

    // Convolve in Fourier space (element-wise multiplication)
    fftw_complex *kernel_fft = (fftw_complex *)grid->smoothing_kernel;
    #pragma omp parallel for
    for (long int i = 0; i < c_size; i++) {
        q_fft[i] *= kernel_fft[i];
    }

    // Perform inverse FFT to get the smoothed charge distribution
    irfft_3d(n, n_loc, q_fft, q);

    free(q_fft);
}

/*
Allocate and initialize the Wendland C2 smoothing kernel in Fourier space.
The kernel is the normalized Wendland C2 screening density (compact support r <= sigma):
    rho(r) propto (1 - r/sigma)^4 * (4*r/sigma + 1), for r <= sigma, 0 otherwise.
It is generated in real space, normalized discretely (as done for the Gaussian kernel) and then
transformed to Fourier space using the forward FFT.
*/
void smooth_charges_wendland_c2_init(grid *grid) {
    int n = grid->n;
    int nh = n / 2 + 1;
    long int n2 = n * n;
    long int c_size = grid->n_local * nh * n;  // Size of the complex-space grid for the local portion
    long int r_size = grid->n_local * n2;  // Size of the real-space grid for the local portion

    double sigma = grid->smoothing_sigma / grid->h;  // Convert sigma to grid units

    double *wendland_kernel = (double *)calloc(r_size, sizeof(double));

    // Generate the Wendland C2 kernel in real space
    int i, j, k;
    int di, dj, dk;
    double ri, rj, r2, r, t;
    for (int i_loc = 0; i_loc < grid->n_local; i_loc++) {
        i = grid->n_start + i_loc;
        di = i > n / 2 ? i - n : i;  // Wrap around for periodicity
        ri = di * di;
        for (j = 0; j < n; j++) {
            dj = j > n / 2 ? j - n : j;  // Wrap around for periodicity
            rj = ri + dj * dj;
            for (k = 0; k < n; k++) {
                dk = k > n / 2 ? k - n : k;  // Wrap around for periodicity
                r2 = rj + dk * dk;
                r = sqrt(r2);
                if (r <= sigma) {
                    t = r / sigma;
                    wendland_kernel[i_loc * n2 + j * n + k] = pow(1.0 - t, 4) * (4.0 * t + 1.0);
                }
            }
        }
    }

    // Normalize the kernel
    double sum = 0.0;
    for (i = 0; i < r_size; i++) {
        sum += wendland_kernel[i];
    }
    allreduce_sum(&sum, 1);
    dscal(wendland_kernel, 1.0 / sum, r_size);  // Normalize so that the sum of the kernel is 1
    dscal(wendland_kernel, 1.0 / pow(n, 3), r_size);  // FFT normalization factor

    // Convert the kernel in Fourier space
    grid->smoothing_kernel = malloc(c_size * sizeof(fftw_complex));
    rfft_3d(n, grid->n_local, wendland_kernel, (fftw_complex *)grid->smoothing_kernel);

    free(wendland_kernel);
}

/*
Allocate and initialize the Wendland C4 smoothing kernel in Fourier space.
The kernel is the normalized Wendland C4 screening density (compact support r <= sigma):
    rho(r) propto (1 - r/sigma)^6 * (35*(r/sigma)^2 + 18*(r/sigma) + 3), for r <= sigma, 0 otherwise.
Same generation/normalization/FFT scheme as the Wendland C2 kernel above.
*/
void smooth_charges_wendland_c4_init(grid *grid) {
    int n = grid->n;
    int nh = n / 2 + 1;
    long int n2 = n * n;
    long int c_size = grid->n_local * nh * n;  // Size of the complex-space grid for the local portion
    long int r_size = grid->n_local * n2;  // Size of the real-space grid for the local portion

    double sigma = grid->smoothing_sigma / grid->h;  // Convert sigma to grid units

    double *wendland_kernel = (double *)calloc(r_size, sizeof(double));

    // Generate the Wendland C4 kernel in real space
    int i, j, k;
    int di, dj, dk;
    double ri, rj, r2, r, t;
    for (int i_loc = 0; i_loc < grid->n_local; i_loc++) {
        i = grid->n_start + i_loc;
        di = i > n / 2 ? i - n : i;  // Wrap around for periodicity
        ri = di * di;
        for (j = 0; j < n; j++) {
            dj = j > n / 2 ? j - n : j;  // Wrap around for periodicity
            rj = ri + dj * dj;
            for (k = 0; k < n; k++) {
                dk = k > n / 2 ? k - n : k;  // Wrap around for periodicity
                r2 = rj + dk * dk;
                r = sqrt(r2);
                if (r <= sigma) {
                    t = r / sigma;
                    wendland_kernel[i_loc * n2 + j * n + k] = pow(1.0 - t, 6) * (35.0 * t * t + 18.0 * t + 3.0);
                }
            }
        }
    }

    // Normalize the kernel
    double sum = 0.0;
    for (i = 0; i < r_size; i++) {
        sum += wendland_kernel[i];
    }
    allreduce_sum(&sum, 1);
    dscal(wendland_kernel, 1.0 / sum, r_size);  // Normalize so that the sum of the kernel is 1
    dscal(wendland_kernel, 1.0 / pow(n, 3), r_size);  // FFT normalization factor

    // Convert the kernel in Fourier space
    grid->smoothing_kernel = malloc(c_size * sizeof(fftw_complex));
    rfft_3d(n, grid->n_local, wendland_kernel, (fftw_complex *)grid->smoothing_kernel);

    free(wendland_kernel);
}

void smooth_charges_diffusion(grid *grid) {
    int n = grid->n;
    int n_loc = grid->n_local;
    int n_start = grid->n_start;

    long int i, j, k;
    long int i0, i1, i2;
    long int j0, j1, j2;
    long int k1, k2;
    long int n2 = n * n;
    long int size = grid->size;

    // Precompute neighbor indices for periodic BCs in j and k
    int jprev[n];
    int jnext[n];
    int kprev[n];
    int knext[n];
    for (int t = 0; t < n; ++t) {
        kprev[t] = ((t - 1 + n ) % n);
        knext[t] = ((t + 1      ) % n);
        jprev[t] = kprev[t] * n;
        jnext[t] = knext[t] * n;
    }

    double D = 1 / 6.2;  // Diffusion coefficient for a simple 3D diffusion process on a grid
    double sigma = grid->smoothing_sigma / grid->h;  // Convert sigma to grid units
    // Keep the diffusion kernel independent of charge-window deconvolution.
    int num_steps = (int)lround(sigma * sigma / (2.0 * D));
    if (num_steps < 1) num_steps = 1;

    double *u = grid->q;  // Input charge distribution
    double *u_new = (double *)malloc(size * sizeof(double));  // Temporary array for the new charge distribution
    vec_copy(u, u_new, size);  // Initialize the new charge distribution with the current values

    for (int step = 0; step < num_steps; step++) {
        // Exchange the top and bottom slices
        mpi_grid_exchange_bot_top(grid->q, n_loc, n);

        #pragma omp parallel for private(i, j, k, i0, i1, i2, j0, j1, j2, k1, k2)
        for (i = 0; i < n_loc; i++) {
            i0 = i * n2;
            i1 = i0 + n2;
            i2 = i0 - n2;
            for (j = 0; j < n; j++) {
                j0 = j * n;
                j1 = jnext[j];
                j2 = jprev[j];
                for (k = 0; k < n; k++) {
                    k1 = knext[k];
                    k2 = kprev[k];
                    u_new[i0 + j0 + k] += D * (
                        u[i1 + j0 + k] +
                        u[i2 + j0 + k] +
                        u[i0 + j1 + k] +
                        u[i0 + j2 + k] +
                        u[i0 + j0 + k1] +
                        u[i0 + j0 + k2] -
                        u[i0 + j0 + k] * 6.0
                    );
                }
            }
        }

        vec_copy(u_new, u, size);  // Copy the new charge distribution back to the original array
    }

    free(u_new);
}

/*
Deconvolve the leading-order charge-assignment window from the mesh density.

Assignment and interpolation contribute W_hat(k)^2. For a B-spline of order P,

    W_hat(k)^2 = 1 - P (k h)^2 / 12 + O((k h)^4).

The separable filter T = 1 - (P/12) delta^2 has the inverse transfer function
to leading order. Its one-dimensional taps are [-b, 1+2b, -b], b=P/12, and
sum to one, so the correction conserves total charge.
*/
void grid_deconvolve_window(grid *grid) {
    const int P = grid->smoothing_window_order;
    if (P <= 0) return;

    const int n = grid->n;
    const int n_loc = grid->n_local;
    const long int n2 = (long int)n * n;
    const long int size = (long int)n_loc * n2;
    const double b = P / 12.0;

    if (size == 0) return;

    double *u = grid->q;
    if (grid->deconv_scratch == NULL) {
        grid->deconv_scratch = (double *)malloc(size * sizeof(double));
        if (grid->deconv_scratch == NULL) {
            mpi_fprintf(stderr, "Unable to allocate window deconvolution scratch\n");
            exit(1);
        }
    }
    double *tmp = grid->deconv_scratch;

    int prev[n], next[n];
    for (int t = 0; t < n; ++t) {
        prev[t] = (t - 1 + n) % n;
        next[t] = (t + 1) % n;
    }

    // x is split across MPI ranks and requires halo planes.
    mpi_grid_exchange_bot_top(u, n_loc, n);
    #pragma omp parallel for
    for (int i = 0; i < n_loc; ++i) {
        const long int i0 = (long int)i * n2;
        for (long int t = 0; t < n2; ++t) {
            tmp[i0 + t] = u[i0 + t]
                        - b * (u[i0 - n2 + t] - 2.0 * u[i0 + t] + u[i0 + n2 + t]);
        }
    }

    // y and z are local to each rank and wrap periodically.
    #pragma omp parallel for
    for (int i = 0; i < n_loc; ++i) {
        const long int i0 = (long int)i * n2;
        for (int j = 0; j < n; ++j) {
            const long int j0 = i0 + (long int)j * n;
            const long int jm = i0 + (long int)prev[j] * n;
            const long int jp = i0 + (long int)next[j] * n;
            for (int k = 0; k < n; ++k) {
                u[j0 + k] = tmp[j0 + k]
                          - b * (tmp[jm + k] - 2.0 * tmp[j0 + k] + tmp[jp + k]);
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < n_loc; ++i) {
        const long int i0 = (long int)i * n2;
        for (int j = 0; j < n; ++j) {
            const long int j0 = i0 + (long int)j * n;
            for (int k = 0; k < n; ++k) {
                tmp[j0 + k] = u[j0 + k]
                            - b * (u[j0 + prev[k]] - 2.0 * u[j0 + k] + u[j0 + next[k]]);
            }
        }
    }
    vec_copy(tmp, u, size);
}

void grid_smoothing_init(grid *grid, int method, double r_cut, double sigma, int window_order) {
    grid->smoothing = method;
    grid->smoothing_rcut = r_cut;
    grid->smoothing_sigma = sigma;
    grid->smoothing_window_order = window_order;
    grid->deconv_scratch = NULL;
    grid->smoothing_kernel = NULL;  // Initialize the smoothing kernel to NULL

    switch (grid->smoothing) {
        case SMOOTHING_TYPE_NONE:
            grid->smooth_charges = smooth_charges_none;
            break;
        case SMOOTHING_TYPE_GAUSS:
            // For now performed outside in theh python code
            smooth_charges_gauss_init(grid);  // Initialize the Gaussian smoothing kernel if needed
            grid->smooth_charges = smooth_charges_fourier_kernel;
            if (grid->smoothing_rcut <= 0.0 || grid->smoothing_sigma <= 0.0) {
                mpi_fprintf(stderr, "Invalid parameters for Gaussian smoothing:\n");
                mpi_fprintf(stderr, "r_cut: %f, sigma: %f\n", grid->smoothing_rcut, grid->smoothing_sigma);
                exit(1);
            }
            break;
        case SMOOTHING_TYPE_WENDLAND_C2:
            // The Wendland C2 screening density has exact compact support r <= sigma, so sigma is
            // the only meaningful length scale: the short-range cutoff is forced to match it
            // regardless of whatever r_cut value was supplied.
            if (grid->smoothing_sigma <= 0.0) {
                mpi_fprintf(stderr, "Invalid parameters for Wendland C2 smoothing:\n");
                mpi_fprintf(stderr, "sigma: %f\n", grid->smoothing_sigma);
                exit(1);
            }
            grid->smoothing_rcut = grid->smoothing_sigma;
            smooth_charges_wendland_c2_init(grid);
            grid->smooth_charges = smooth_charges_fourier_kernel;
            break;
        case SMOOTHING_TYPE_WENDLAND_C4:
            // Same reasoning as Wendland C2: exact compact support r <= sigma, so sigma is the
            // only meaningful length scale for the short-range cutoff too.
            if (grid->smoothing_sigma <= 0.0) {
                mpi_fprintf(stderr, "Invalid parameters for Wendland C4 smoothing:\n");
                mpi_fprintf(stderr, "sigma: %f\n", grid->smoothing_sigma);
                exit(1);
            }
            grid->smoothing_rcut = grid->smoothing_sigma;
            smooth_charges_wendland_c4_init(grid);
            grid->smooth_charges = smooth_charges_fourier_kernel;
            break;
        case SMOOTHING_TYPE_WENDLAND_C2_NOFFT:
        case SMOOTHING_TYPE_WENDLAND_C4_NOFFT:
            if (grid->smoothing_sigma <= 0.0) {
                mpi_fprintf(stderr, "Invalid parameters for no-FFT Wendland smoothing:\n");
                mpi_fprintf(stderr, "sigma: %f\n", grid->smoothing_sigma);
                exit(1);
            }
            grid->smoothing_rcut = grid->smoothing_sigma;
            smooth_charges_wendland_nofft_init(
                grid,
                grid->smoothing == SMOOTHING_TYPE_WENDLAND_C2_NOFFT ? 2 : 4
            );
            grid->smooth_charges = smooth_charges_wendland_nofft;
            break;
        case SMOOTHING_TYPE_WENDLAND_C2_POLY:
        case SMOOTHING_TYPE_WENDLAND_C4_POLY:
            if (grid->smoothing_sigma <= 0.0) {
                mpi_fprintf(stderr, "Invalid parameters for polynomial Wendland smoothing:\n");
                mpi_fprintf(stderr, "sigma: %f\n", grid->smoothing_sigma);
                exit(EXIT_FAILURE);
            }
            grid->smoothing_rcut = grid->smoothing_sigma;
            smooth_charges_wendland_poly_init(grid);
            grid->smooth_charges = smooth_charges_wendland_poly;
            break;
        case SMOOTHING_TYPE_DIFFUSION:
            grid->smooth_charges = smooth_charges_diffusion;
            if (
                grid->smoothing_rcut <= 0.0 || grid->smoothing_sigma <= 0.0
            ) {
                mpi_fprintf(stderr, "Invalid parameters for diffusion-based smoothing:\n");
                mpi_fprintf(stderr, "r_cut: %f, sigma: %f\n", grid->smoothing_rcut, grid->smoothing_sigma);
                exit(1);
            }
            break;
        // Additional smoothing methods can be added here
        default:
            break;
    }

    // Additional initialization for smoothing can be added here if needed
}

void grid_smoothing_free(grid *grid) {
    if (grid->deconv_scratch != NULL) {
        free(grid->deconv_scratch);
        grid->deconv_scratch = NULL;
    }

    if (
        grid->smoothing == SMOOTHING_TYPE_WENDLAND_C2_NOFFT ||
        grid->smoothing == SMOOTHING_TYPE_WENDLAND_C4_NOFFT
    ) {
        smooth_charges_wendland_nofft_free(grid);
        return;
    }
    if (
        grid->smoothing == SMOOTHING_TYPE_WENDLAND_C2_POLY ||
        grid->smoothing == SMOOTHING_TYPE_WENDLAND_C4_POLY
    ) {
        smooth_charges_wendland_poly_free(grid);
        return;
    }
    if (grid->smoothing_kernel != NULL) {
        free(grid->smoothing_kernel);
        grid->smoothing_kernel = NULL;
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
    grid_smoothing_free(grid);

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

void grid_update_eps_and_k2_sphere(grid *g, particles *p)
{
    const int n       = g->n;
    const int n_local = g->n_local;
    const int n_start = g->n_start;
    const double h    = g->h;
    const long size   = g->size;
    const double L    = g->L;

    const double eps_s = g->eps_s;
    const double eps_m = g->eps_int;
    const double kbar2 = g->kbar2;

    double       *eps_x  = g->eps_x;
    double       *eps_y  = g->eps_y;
    double       *eps_z  = g->eps_z;
    double       *k2     = g->k2;
    unsigned int *region = g->region;

    /* ====================================================
     * STEP 1 - classify inside/outside using VdW spheres
     * ==================================================== */

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

        eps_x[idx] = eps_s;
        eps_y[idx] = eps_s;
        eps_z[idx] = eps_s;

        /* Edge X */
        long idx_px = idx + (long)n * n;
        if (region[idx] != region[idx_px]) {
            double frac = sphere_edge_fraction(p, x1, y1, z1, h, 0, L);
            eps_x[idx] = eps_mix_eval(eps_m, eps_s, frac);
        } else if (region[idx] != 0) {
            eps_x[idx] = eps_m;
        }

        /* Edge Y */
        long idx_py = (j == n - 1) ? idx - (long)(n - 1) * n : idx + n;
        if (region[idx] != region[idx_py]) {
            double frac = sphere_edge_fraction(p, x1, y1, z1, h, 1, L);
            eps_y[idx] = eps_mix_eval(eps_m, eps_s, frac);
        } else if (region[idx] != 0) {
            eps_y[idx] = eps_m;
        }

        /* Edge Z */
        long idx_pz = (k == n - 1) ? idx - (n - 1) : idx + 1;
        if (region[idx] != region[idx_pz]) {
            double frac = sphere_edge_fraction(p, x1, y1, z1, h, 2, L);
            eps_z[idx] = eps_mix_eval(eps_m, eps_s, frac);
        } else if (region[idx] != 0) {
            eps_z[idx] = eps_m;
        }
    }


    /* ====================================================
     * STEP 3 — mark enlarged sphere (region = 2)
    Solvent points (region = 0) inside the integration sphere of
    any particle are marked as 2.
    The radius used is the same as in compute_stress_tensor_forces_spherical:
    R = ceil(solv_radii / h) + 2 (in cell units)
     * ==================================================== */
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
                }
            }
        }
    }

    // Exchange the final region map used by the stress tensor
    mpi_grid_exchange_bot_top_uint(region, n_local, n);
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
