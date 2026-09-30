#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "linalg.h"
#include "verlet.h"
#include "mp_structs.h"
#include "mpi_base.h"

#define JACOBI_OMEGA 0.66

double mg_corr_scale = MG_CORR_SCALE;
int mg_maxdepth = MG_MAXDEPTH;
int mg_keep_y = 0;

void mg_env_init(void) {
    const char *a = getenv("MAZE_MG_CORR_SCALE"), *b = getenv("MAZE_MG_MAXDEPTH"), *c = getenv("MAZE_MG_KEEP_Y");
    if (a != NULL && atof(a) > 0.0) mg_corr_scale = atof(a);
    if (b != NULL && atoi(b) > 0) mg_maxdepth = atoi(b);
    if (c != NULL) mg_keep_y = (atoi(c) != 0);
}

static int cg_coarse(double* b, double* x, int s1, int s2, int maxit, double rtol)
{
    const long n = (long)s1 * (long)s2 * (long)s2;

    /* --- scalari --- */
    int    k;
    double r0_inf, r_inf;
    double r_dot_v, denom, alpha;
    double rn_rn, rn_dot_vn, beta;

    /* --- buffer con ghost --- */
    double *r;
    double *p;
    double *Ap;

    /* --- allocazioni --- */
    r  = mpi_grid_allocate(s1, s2);
    p  = mpi_grid_allocate(s1, s2);
    Ap = mpi_grid_allocate(s1, s2);

    /* r = A x - b  (x è il guess iniziale passatoci dal chiamante) */
    // mpi_grid_exchange_bot_top(x, s1, s2);
    laplace_filter(x, r, s1, s2);   /* r <- A x */
    daxpy(b, r, -1.0, n);           /* r <- r - b */

    /* p = -v = r/6  con v = -r/6 (precondizionatore diagonale del Laplaciano) */
    memcpy(p, r, n * sizeof(double));
    dscal(p, 1.0/6.0, n);

    /* norma iniziale (stop relativo) */
    r0_inf = norm_inf(r, n);
    if (r0_inf == 0.0) {
        mpi_grid_free(r, s2);
        mpi_grid_free(p, s2);
        mpi_grid_free(Ap, s2);
        return 0;
    }

    /* r_dot_v = <r, v> = -<r, p> perché p = -v */
    r_dot_v = - ddot(r, p, n);

    for (k = 0; k < maxit; ++k) {
        /* Ap = A p */
        // mpi_grid_exchange_bot_top(p, s1, s2);
        laplace_filter(p, Ap, s1, s2);

        denom = ddot(p, Ap, n);
        if (fabs(denom) < 1e-300) {
            /* direzione quasi nulla / breakdown */
            break;
        }

        alpha = r_dot_v / denom;

        /* x <- x + alpha p */
        daxpy(p, x, alpha, n);

        /* r <- r + alpha Ap   (perché r = A x - b) */
        daxpy(Ap, r, alpha, n);

        /* criterio di arresto relativo in norma infinito */
        r_inf = norm_inf(r, n);
        if (r_inf <= rtol * r0_inf) { // TODO: questo non lo capisco
            ++k;           /* conta anche l’iterazione corrente */
            break;
        }

        /* update scalari */
        rn_rn     = ddot(r, r, n);
        rn_dot_vn = - rn_rn / 6.0;     /* <r_new, v_new> con v_new = -r_new/6 */
        beta      = rn_dot_vn / r_dot_v;
        r_dot_v   = rn_dot_vn;

        /* p <- beta p + r/6 */
        dscal(p, beta, n);
        daxpy(r, p, 1.0/6.0, n);
    }

    mpi_grid_free(r,  s2);
    mpi_grid_free(p,  s2);
    mpi_grid_free(Ap, s2);

    return k;  /* numero di iterazioni eseguite */
}


int v_cycle(double *in, double *out, int s1, int s2, int n_start, int sm, int depth) {
    int res;
    const long int n2   = (long int)s2 * s2;
    const long int size = (long int)s1 * n2;

    // next level sizes/parity
    const int s1_nxt      = (s1 + 1 - (n_start % 2)) / 2;
    const int s2_nxt      = s2 / 2;

    // Se lo giri senza MPI funziona anche senza il +1 ma con il +1 server per tenero conto di quando n_start e' dispari
    const int n_start_nxt = (n_start + 1) / 2;   // CHANGED: floor  

    const int sm_iter = (int)ceil(sm * pow(MG_RECURSION_FACTOR, depth));

    // base case
    if ( (s1_nxt < fmax(16, get_size())) || (depth >= mg_maxdepth) ) {
        if (depth == 0) {
            mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
            mpi_fprintf(stderr, "Multigrid: requires atleast one level of recursion (s1 >= %d)\n", fmax(4, get_size()));
            mpi_fprintf(stderr, "Increase the size of the grid or reduce the number of MPI processes.\n");
            mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
            exit(1);
        }
        // smooth(in, out, s1, s2, sm_iter);
        // printf("Solving exact at depth %d with CG\n", depth);
        cg_coarse(in, out, s1, s2, 50, 1e-5); //prima era 1e-4
        // conj_grad(in, out, out, 1e-5, s1, s2);
        return depth;
    }

    const long int size_nxt = (long int)s1_nxt * s2_nxt * s2_nxt;

    // buffers
    double *r   = mpi_grid_allocate(s1, s2);
    double *rhs = mpi_grid_allocate(s1_nxt, s2_nxt);
    double *eps = mpi_grid_allocate(s1_nxt, s2_nxt);

    memset(eps, 0, size_nxt * sizeof(double));

    // 1) pre-smoothing
    smooth(in, out, s1, s2, sm_iter);

    // 2) residual: r = in - A*out
    laplace_filter(out, r, s1, s2);
    dscal(r, -1.0, size);
    daxpy(in, r, 1.0, size);

    // 3) restrict to coarse
    // mpi_grid_exchange_bot_top(r, s1, s2);                 // CHANGED
    restriction(r, rhs, s1, s2, n_start);

    // 4) coarse solve (recursive)
    res = v_cycle(rhs, eps, s1_nxt, s2_nxt, n_start_nxt, sm, depth + 1);

    // 5) prolong correction
    // mpi_grid_exchange_bot_top(eps, s1_nxt, s2_nxt);       // CHANGED
    prolong(eps, r, s1_nxt, s2_nxt, s1, s2, n_start);

    // 6) apply correction
    daxpy(r, out, mg_corr_scale, size);

    // 7) post-smoothing
    smooth(in, out, s1, s2, sm_iter);

    mpi_grid_free(r, s2);
    mpi_grid_free(rhs, s2_nxt);
    mpi_grid_free(eps, s2_nxt);

    return res;
}

int multigrid_apply_recursive(double *in, double *out, int s1, int s2, int n_start1, int sm) {
    // memset(out, 0, s1 * s2 * s2 * sizeof(double));  // Initialize out to zero
    return v_cycle(in, out, s1, s2, n_start1, sm, 0);  // Apply the recursive V-cycle multigrid method
}

/*
Apply the multigrid method to solve the Poisson equation  A.out = in
using a 3-level V-cycle multigrid method.

@param in: input array (right-hand side of the equation)
@param out: in/out array (starting guess/solution)
@param s1: size of the first dimension (number of slices)
@param s2: size of the second dimension (number of grid points per slice)
@param n_start1: starting index for the first dimension (used for restriction)
*/
void multigrid_apply_3lvl(double *in, double *out, int s1, int s2, int n_start1, int sm) {
    int n1 = s2;
    int n2 = n1 / 2;
    int n3 = n2 / 2;

    int n_loc1 = s1;

    int n_loc2 = (n_loc1 + 1 - (n_start1 % 2)) / 2;
    int n_start2 = (n_start1 + 1) / 2;

    int n_loc3 = (n_loc2 + 1 - (n_start2 % 2)) / 2;
    // int n_start3 = (n_start2 + 1) / 2;

    if (n_loc3 == 0) {
        mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
        mpi_fprintf(stderr, "Warning: after restriction some processors have no local grid points!\n");
        mpi_fprintf(stderr, "This case is not yet implemented, please use MULTIGRID with atleast 4 slices\n");
        mpi_fprintf(stderr, "per processor (N_grid / num_mpi_procs >= 4) \n");
        mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
        exit(1);
    }

    long int size1 = n_loc1 * n1 * n1;
    long int size2 = n_loc2 * n2 * n2;
    long int size3 = n_loc3 * n3 * n3;

    double *r1 = mpi_grid_allocate(n_loc1, n1);
    double *r2 = mpi_grid_allocate(n_loc2, n2);
    double *r3 = mpi_grid_allocate(n_loc3, n3);
    double *e2 = mpi_grid_allocate(n_loc2, n2);
    double *e3 = mpi_grid_allocate(n_loc3, n3);
    double *tmp2 = mpi_grid_allocate(n_loc2, n2);

    // memset(out, 0, size1 * sizeof(double));  // out = 0
    memset(e2, 0, size2 * sizeof(double));  // e2 = 0
    memset(e3, 0, size3 * sizeof(double));  // e3 = 0

    smooth(in, out, n_loc1, n1, sm);  // out = smooth(in, out)  ~solve(A . out = in)
    // r1  =  in - A . out
    laplace_filter(out, r1, n_loc1, n1);
    dscal(r1, -1.0, size1);
    daxpy(in, r1, 1.0, size1);
    restriction(r1, r2, n_loc1, n1, n_start1);  // r2 = restriction(r1)

    smooth(r2, e2, n_loc2, n2, (int)ceil(sm * 1.2));  // e2 = smooth(r2)  ~solve(A . e2 = r2)
    // tmp2  =  r2 - A . e2
    laplace_filter(e2, tmp2, n_loc2, n2);
    dscal(tmp2, -1.0, size2);
    daxpy(r2, tmp2, 1.0, size2);
    restriction(tmp2, r3, n_loc2, n2, n_start2);  // r3 = restriction(r2 - A . e2)


    smooth(r3, e3, n_loc3, n3, (int)ceil(sm * 1.5));  // e3 = smooth(r3)  ~solve(A . e3 = r3)
    prolong(e3, r2, n_loc3, n3, n_loc2, n2, n_start2);
    daxpy(r2, e2, 1.0, size2);  // e2 = e2 + prolong(e3)

    smooth(r2, e2, n_loc2, n2, (int)ceil(sm * 1.2));  // e2 = smooth(r2, e2)  ~solve(A . e2 = r2)
    prolong(e2, r1, n_loc2, n2, n_loc1, n1, n_start1);
    daxpy(r1, out, 1.0, size1);  // out = out + prolong(e2)

    smooth(in, out, n_loc1, n1, sm);  // out = smooth(in, out)  ~solve(A . out = in)

    mpi_grid_free(r1, n1);
    mpi_grid_free(r2, n2);
    mpi_grid_free(r3, n3);
    mpi_grid_free(e2, n2);
    mpi_grid_free(e3, n3);
    mpi_grid_free(tmp2, n2);
}

/*
Apply the multigrid method to solve the Poisson equation  A.out = in
using a 3-level V-cycle multigrid method.

@param in: input array (right-hand side of the equation)
@param out: in/out array (starting guess/solution)
@param s1: size of the first dimension (number of slices)
@param s2: size of the second dimension (number of grid points per slice)
@param n_start1: starting index for the first dimension (used for restriction)
*/
void multigrid_apply_2lvl(double *in, double *out, int s1, int s2, int n_start1, int sm) {
    int n1 = s2;
    int n2 = n1 / 2;

    int n_loc1 = s1;

    int n_loc2 = (n_loc1 + 1 - (n_start1 % 2)) / 2;
    int n_start2 = (n_start1 + 1) / 2;


    if (n_loc2 == 0) {
        mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
        mpi_fprintf(stderr, "Warning: after restriction some processors have no local grid points!\n");
        mpi_fprintf(stderr, "This case is not yet implemented, please use MULTIGRID with atleast 4 slices\n");
        mpi_fprintf(stderr, "per processor (N_grid / num_mpi_procs >= 4) \n");
        mpi_fprintf(stderr, "------------------------------------------------------------------------------------\n");
        exit(1);
    }

    long int size1 = n_loc1 * n1 * n1;
    long int size2 = n_loc2 * n2 * n2;

    double *r1 = mpi_grid_allocate(n_loc1, n1);
    double *r2 = mpi_grid_allocate(n_loc2, n2);
    double *e2 = mpi_grid_allocate(n_loc2, n2);
    double *tmp2 = mpi_grid_allocate(n_loc2, n2);

    // memset(out, 0, size1 * sizeof(double));  // out = 0
    memset(e2, 0, size2 * sizeof(double));  // e2 = 0

    smooth(in, out, n_loc1, n1, sm);  // out = smooth(in, out)  ~solve(A . out = in)
    // r1  =  in - A . out
    laplace_filter(out, r1, n_loc1, n1);
    dscal(r1, -1.0, size1);
    daxpy(in, r1, 1.0, size1);
    restriction(r1, r2, n_loc1, n1, n_start1);  // r2 = restriction(r1)


    smooth(r2, e2, n_loc2, n2, (int)ceil(sm * 1.2));  // e2 = smooth(r2, e2)  ~solve(A . e2 = r2)
    prolong(e2, r1, n_loc2, n2, n_loc1, n1, n_start1);
    daxpy(r1, out, 1.0, size1);  // out = out + prolong(e2)

    smooth(in, out, n_loc1, n1, sm);  // out = smooth(in, out)  ~solve(A . out = in)

    mpi_grid_free(r1, n1);
    mpi_grid_free(r2, n2);
    mpi_grid_free(e2, n2);
    mpi_grid_free(tmp2, n2);
}

void prolong_nearestneighbors(
    double *in, double *out, int s1, int s2,
    int target_s1, int target_s2, int target_n_start
) {
    int a, b;
    long int i, j, k;
    long int i0, j0, k0;
    long int i1, j1, k1;
    long int n2 = s2 * s2;

    long int target_n2 = target_s2 * target_s2;

    double app;

    int d = target_n_start % 2;
    double should_exchange = d;

    #pragma omp parallel for private(i, j, k, i0, j0, k0, i1, j1, k1, a, b, app)
    for (i = 0; i < s1; i++) {
        a = i * n2;
        i0 = (i * 2 + d) * target_n2;
        i1 = i0 + target_n2;
        for (j = 0; j < s2; j++) {
            b = j * s2;
            j0 = j * 2 * target_s2;
            j1 = j0 + target_s2;
            for (k = 0; k < s2; k++) {
                k0 = k * 2;
                k1 = k0 + 1;
                app = in[a + b + k] * 0.125;
                out[i0 + j0 + k0] = app;
                out[i0 + j0 + k1] = app;
                out[i0 + j1 + k0] = app;
                out[i0 + j1 + k1] = app;
                out[i1 + j0 + k0] = app;
                out[i1 + j0 + k1] = app;
                out[i1 + j1 + k0] = app;
                out[i1 + j1 + k1] = app;
            }
        }
    }

    allreduce_max(&should_exchange, 1);
    if (should_exchange) {
        // In case of odd number of slices, we need to wrap around the top slice from the proc below
        // as the 1st bottom slice of the current proc
        mpi_grid_exchange_bot_top(out, target_s1, target_s2);  // Called outside the if to avoid deadlock
        if (d) {
            vec_copy(out - target_n2, out, target_n2);
        }
    }
}

void calc_w0_w1(int cond, double *w0, double *w1) {
    if (cond) {
        *w0 = 0.5;
        *w1 = 0.5;
    } else {
        *w0 = 1.0;
        *w1 = 0.0;
    }
}

/*
Apply trilinear interpolation prolongation to transfer corrections
from a coarse grid (s1 x s2 x s2) to a fine grid (target_s1 x target_s2 x target_s2).
Each fine node is computed once using the 8 neighboring coarse nodes.
Periodic BCs are applied in j,k. Along i we rely on a single halo exchange
to read I+1 on rank boundaries. No writes to halo memory are performed here.

@param in:            input coarse-grid array
@param out:           output fine-grid array (overwritten; interior only)
@param s1:            coarse number of slices in i
@param s2:            coarse number of points per line in j,k
@param target_s1:     fine number of slices in i
@param target_s2:     fine number of points per line in j,k
@param target_n_start: starting index (global) for the fine grid in i (parity across MPI ranks)
*/
void prolong_trilinear(
    double *in, double *out, int s1, int s2,
    int target_s1, int target_s2, int target_n_start
) {
    long int n2c = s2 * s2;
    long int n2f = target_s2 * target_s2;

    long int row_f;
    long int i0, j0, k0;
    long int i1, j1, k1;
    double wi0, wj0, wk0;
    double wi1, wj1, wk1;

    int d = target_n_start % 2;
    double should_exchange = d;

    // Precompute neighbor indices for periodic BCs in j and k
    long int jnext[s2];
    long int knext[s2];
    for (int t = 0; t < s2; ++t) {
        knext[t] = ((t + 1) % s2);
        jnext[t] = knext[t] * s2;
    }

    mpi_grid_exchange_bot_top(in, s1, s2);

    #pragma omp parallel for private(i0, j0, k0, i1, j1, k1, row_f, wi0, wj0, wk0, wi1, wj1, wk1)
    for (int i = 0; i < target_s1; i++) {
        i0 = (i < d) ? (s1 - 1) : ((i - d) / 2);
        i0 = i0 * n2c;
        i1 = i0 + n2c;

        calc_w0_w1((i ^ d) % 2, &wi0, &wi1);
        for (int j = 0; j < target_s2; j++) {
            j0 = ((j / 2    ) % s2);
            j1 = jnext[j0];
            j0 *= s2;

            row_f = i * n2f + j * target_s2;

            calc_w0_w1(j % 2, &wj0, &wj1); 
            for (int k = 0; k < target_s2; k++) {
                k0 = (k / 2    ) % s2;
                k1 = knext[k0];

                calc_w0_w1(k % 2, &wk0, &wk1);
                out[row_f + k] = (
                    in[i0 + j0 + k0] * wi0 * wj0 * wk0 +
                    in[i1 + j0 + k0] * wi1 * wj0 * wk0 +
                    in[i0 + j1 + k0] * wi0 * wj1 * wk0 +
                    in[i1 + j1 + k0] * wi1 * wj1 * wk0 +
                    in[i0 + j0 + k1] * wi0 * wj0 * wk1 +
                    in[i1 + j0 + k1] * wi1 * wj0 * wk1 +
                    in[i0 + j1 + k1] * wi0 * wj1 * wk1 +
                    in[i1 + j1 + k1] * wi1 * wj1 * wk1
                );
            }
        }
    }

    allreduce_max(&should_exchange, 1);
    if (should_exchange > 0.5) {
        // In case of odd number of slices, we need to wrap around the top slice from the proc below
        // as the 1st bottom slice of the current proc
        mpi_grid_exchange_bot_top(out, target_s1, target_s2);  // Called outside the if to avoid deadlock
        if (d) {
            vec_copy(out - n2f, out, n2f);
        }

    }
}


void restriction_8pt(double *in, double *out, int s1, int s2, int n_start) {
    int a, b;
    long int i, j, k;
    long int i0, j0;
    long int i1, j1, k1;
    long int n2 = s2 * s2;

    int s3 = s2 / 2;
    long int n3 = s3 * s3;

    double should_exchange = s1 % 2;

    // If the number of slices in the first dimension is odd, we need to wrap around
    // the bottom slice above to apply the averaging with PBCs
    allreduce_max(&should_exchange, 1);
    if (should_exchange > 0.5) {
        mpi_grid_exchange_bot_top(in, s1, s2);
    }   

    #pragma omp parallel for private(i, j, k, i0, i1, j0, j1, k1, a, b)
    for (i = n_start % 2; i < s1; i+=2) {
        i0 = i * n2;
        i1 = (i+1) * n2;
        a = i / 2 * n3;
        for (j = 0; j < s2; j+=2) {
            j0 = j * s2;
            j1 = ((j+1) % s2) * s2;
            b = j / 2 * s3;
            for (k = 0; k < s2; k+=2) {
                k1 = (k+1) % s2;
                out[a + b + k / 2] = (
                    in[i0 + j0 + k] +
                    in[i0 + j0 + k1] +
                    in[i0 + j1 + k] +
                    in[i0 + j1 + k1] +
                    in[i1 + j0 + k] +
                    in[i1 + j0 + k1] +
                    in[i1 + j1 + k] +
                    in[i1 + j1 + k1]
                ) * 0.125;
            }
        }
    }
}

/*
Apply 27-point full-weighting restriction (nodal) to transfer residuals
from a fine grid (s1 x s2 x s2) to a coarse grid (s1/2 x s2/2 x s2/2).
The coarse value at (I,J,K) corresponds to the fine node (i=2I+offset, j=2J, k=2K),
and is computed with weights: center=8, faces=4, edges=2, corners=1, normalized by 64.

@param in:       input fine-grid array (e.g., residual on fine grid)
@param out:      output coarse-grid array (restricted residual)
@param s1:       number of slices in i (fine grid)
@param s2:       number of points per line in j,k (fine grid)
@param n_start:  global starting index in i (used to preserve parity across MPI ranks)
*/
void restriction_27pt(double *in, double *out, int s1, int s2, int n_start) {
    int s3 = s2 / 2;
    long int n2 = s2 * s2;
    long int n3 = s3 * s3;
    double inv64 = 1.0 / 64.0;

    long int a, b, c;
    long int i0, j0, k0;
    long int i1, j1, k1;
    long int i2, j2, k2;
    double f_sum, e_sum, c_sum;

    // Precompute neighbor indices for periodic BCs in j and k
    int jprev[s2];
    int jnext[s2];
    int kprev[s2];
    int knext[s2];
    for (int t = 0; t < s2; ++t) {
        kprev[t] = ((t - 1 + s2 ) % s2);
        knext[t] = ((t + 1      ) % s2);
        jprev[t] = kprev[t] * s2;
        jnext[t] = knext[t] * s2;
    }

    // If the number of slices in the first dimension is odd, we need to wrap around
    // the bottom slice above to apply the averaging with PBCs
    mpi_grid_exchange_bot_top(in, s1, s2);

    int i_offset = n_start % 2;

    #pragma omp parallel for private(i0, i1, i2, j0, j1, j2, k0, k1, k2, a, b, c, f_sum, e_sum, c_sum)
    for (int i = i_offset; i < s1; i += 2) {
        i0 = i * n2;
        i1 = i0 + n2;      // contiguous ghost
        i2 = i0 - n2;      // contiguous ghost
        a  = (i / 2) * n3;

        for (int j = 0; j < s2; j += 2) {
            j0 = j * s2;
            j1 = jnext[j];
            j2 = jprev[j];
            b = (j / 2) * s3;

            for (int k = 0; k < s2; k += 2) {
                k2 = kprev[k];
                k1 = knext[k];
                c = k / 2;

                f_sum =
                    in[i2 + j0 + k]  + in[i1 + j0 + k] +
                    in[i0 + j2 + k]  + in[i0 + j1 + k] +
                    in[i0 + j0 + k2] + in[i0 + j0 + k1];

                e_sum =
                    in[i1 + j0 + k2] + in[i1 + j0 + k1] +
                    in[i2 + j0 + k2] + in[i2 + j0 + k1] +
                    in[i1 + j2 + k]  + in[i1 + j1 + k] +
                    in[i2 + j2 + k]  + in[i2 + j1 + k] +
                    in[i0 + j2 + k2] + in[i0 + j2 + k1] +
                    in[i0 + j1 + k2] + in[i0 + j1 + k1];

                c_sum =
                    in[i1 + j1 + k2] + in[i1 + j1 + k1] +
                    in[i1 + j2 + k2] + in[i1 + j2 + k1] +
                    in[i2 + j1 + k2] + in[i2 + j1 + k1] +
                    in[i2 + j2 + k2] + in[i2 + j2 + k1];

                out[a + b + c] = (8.0 * in[i0 + j0 + k] + 4.0 * f_sum + 2.0 * e_sum + c_sum) * inv64;
            }
        }
    }
}


/*
Apply the Jacobi smoothing method to solve the Poisson equation.
Gives an approximate solution to the equation A.out = in based

@param in: input array (right-hand side of the equation)
@param out: in/out array (starting guess/solution)
@param s1: size of the first dimension (number of slices)
@param s2: size of the second dimension (number of grid points per slice)
@param tol: number of iterations to perform
*/
void smooth_jacobi(double *in, double *out, int s1, int s2, double tol) {
    long int n3 = s1 * s2 * s2;

    double omega = JACOBI_OMEGA / -6.0;

    double *tmp = (double *)malloc(n3 * sizeof(double));

    for (int iter=0; iter < tol; iter++) { 
        // out = out + omega * (in - A. out)
        laplace_filter(out, tmp, s1, s2);  // res = A . out
        daxpy(tmp, out, -omega, n3);
        daxpy(in, out, omega, n3);
    }

    free(tmp);
}


/*
Apply the Red-Black Gauss-Seidel smoothing method to solve the Poisson equation.
Updates the solution in-place, giving an approximate solution to A . out = in
with a 7-point Laplacian stencil under periodic boundary conditions in j,k.

@param in:   input array (right-hand side of the equation)
@param out:  in/out array (initial guess and updated solution)
@param s1:   size of the first dimension (number of slices in i)
@param s2:   size of the second dimension (number of grid points in j,k)
@param tol:  number of smoothing iterations (each iteration = red + black sweep)
*/
void smooth_rbgs(double *in, double *out, int s1, int s2, double tol) {
    int iters = (int)tol;
    if (iters <= 0) return;

    const long int n2 = s2 * s2;
    const double inv_diag = -1.0 / 6.0;   // diagonal entry of 7-point Laplacian
    const int n_start = get_n_start();    // global offset for MPI parity

    int d;
    long int idx;
    long int i0, j0;
    long int i1, j1, k1;
    long int i2, j2, k2;
    double sum_nb;

    // Precompute neighbor indices for periodic BCs in j and k
    int jprev[s2];
    int jnext[s2];
    int kprev[s2];
    int knext[s2];
    for (int t = 0; t < s2; ++t) {
        kprev[t] = ((t - 1 + s2 ) % s2);
        knext[t] = ((t + 1      ) % s2);
        jprev[t] = kprev[t] * s2;
        jnext[t] = knext[t] * s2;
    }

    for (int iter = 0; iter < iters; ++iter) {
        mpi_grid_exchange_bot_top(out, s1, s2);

        // ----- RED sweep -----
        #pragma omp parallel for private(i0, i1, i2, j0, j1, j2, k1, k2, d, idx, sum_nb)
        for (int i = 0; i < s1; ++i) {
            i0 = i * n2;
            i1 = i0 + n2;
            i2 = i0 - n2;
            d = (n_start + i) % 2;

            for (int j = 0; j < s2; ++j) {
                j0 = j * s2;
                j1 = jnext[j];
                j2 = jprev[j];

                // choose k parity so that (i+j+k) % 2 == 0 (red)
                int k = (2 - ((d + (j % 2)) % 2)) % 2;
                for (; k < s2; k += 2) {
                    k2 = kprev[k];
                    k1 = knext[k];
                    idx = i0 + j0 + k;

                    sum_nb =
                        out[i2 + j0 + k] + out[i1 + j0 + k] +     // i-1, i+1
                        out[i0 + j2 + k] + out[i0 + j1 + k] +     // j-1, j+1
                        out[i0 + j0  + k2] + out[i0 + j0  + k1];  // k-1, k+1

                    out[idx] = (in[idx] - sum_nb) * inv_diag;
                }
            }
        }

        // Exchange again before the black sweep
        mpi_grid_exchange_bot_top(out, s1, s2);

        // ----- BLACK sweep -----
        #pragma omp parallel for private(i0, i1, i2, j0, j1, j2, k1, k2, d, idx, sum_nb)
        for (int i = 0; i < s1; ++i) {
            i0 = i * n2;
            i2 = i0 - n2;
            i1 = i0 + n2;
            d = (n_start + i) % 2;

            for (int j = 0; j < s2; ++j) {
                j0 = j * s2;
                j1 = jnext[j];
                j2 = jprev[j];

                // choose k parity so that (i+j+k) % 2 == 1 (black)
                int k = (1 - ((d + (j % 2)) % 2)) % 2;
                for (; k < s2; k += 2) {
                    k1 = knext[k];
                    k2 = kprev[k];
                    idx = i0 + j0 + k;

                    sum_nb =
                        out[i2 + j0 + k] + out[i1 + j0 + k] +
                        out[i0 + j2 + k] + out[i0 + j1 + k] +
                        out[i0 + j0  + k2] + out[i0 + j0  + k1];

                    out[idx] = (in[idx] - sum_nb) * inv_diag;
                }
            }
        }
    }
}

// void smooth_lcg(double *in, double *out, int s1, int s2, double tol) {
//     conj_grad(in, out, out, 1E-1, s1, s2);  // out = ~solve(A . out = in)
// }


// void smooth_diag(double *in, double *out, int s1, int s2, double tol) {
//     long int n3 = s1 * s2 * s2;
//     for (int i = 0; i < n3; i++) {
//         out[i] = in[i] / -6.0;
//     }
// }

int multigrid_apply(double *in, double *out, int s1, int s2, int n_start1, int sm) {
    multigrid_apply_recursive(in, out, s1, s2, n_start1, sm);
    // multigrid_apply_3lvl(in, out, s1, s2, n_start1, sm);
    // multigrid_apply_2lvl(in, out, s1, s2, n_start1, sm);
}

// choose smoothing, restriction and interpolation
void smooth(double *in, double *out, int s1, int s2, double tol) {
    // smooth_lcg(in, out, s1, s2, tol);
    // smooth_jacobi(in, out, s1, s2, tol);
    // smooth_diag(in, out, s1, s2, tol);
    smooth_rbgs(in, out, s1, s2, tol);
}

void restriction(double *in, double *out, int s1, int s2, int n_start) {
    // restriction_8pt(in, out, s1, s2, n_start);
    restriction_27pt(in, out, s1, s2, n_start);
}

void prolong(double *in, double *out, int s1, int s2, int target_s1, int target_s2, int target_n_start) {
    // prolong_nearestneighbors(in, out, s1, s2, target_s1, target_s2, target_n_start);
    prolong_trilinear(in, out, s1, s2, target_s1, target_s2, target_n_start);
}
