/* [slit_dsl patch] Semi-coarsening multigrid for the PB operator with anisotropic face coefficients.
 *
 * Same fine-grid operator as laplace_filter_pb / smooth_pb_rbgs (index units, periodic):
 *   (A u)_c = sum_faces eps_face (u_nb - u_c) - k2_c u_c.
 * The fine level (cubic s1 = s2) reuses smooth_pb_rbgs and laplace_filter_pb_residual, so the
 * equation solved is unchanged; only the coarse levels differ from multigrid_pb.c:
 *  - each level coarsens a direction only if its mean face coupling is >= 1/2 of the strongest one
 *    (eps = diag(78.5, 78.5, 4): x,y are coarsened, z is kept fine until the couplings equalise);
 *  - coarse faces: series (harmonic) combination along a coarsened direction, full-weighting
 *    average across the others, divided by s_d^2 (index units of the fine grid) -> no correction scale;
 *  - restriction = full weighting, prolongation = (bi)linear interpolation in the coarsened directions;
 *  - coarsest level: CG.
 * Single MPI rank only (the caller falls back to the standard V-cycle otherwise).
 * Grid layout idx = (i*ny + j)*nz + k, x = i, y = j, z = k (as in the rest of the code). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "linalg.h"
#include "mpi_base.h"
#include "laplace.h"
#include "multigrid_pb.h"
#include "multigrid_semi.h"

#define SEMI_MAX_LEVELS 12
#define SEMI_COARSEST_POINTS 4096

typedef struct {
    int n[3];          /* nx, ny, nz */
    int c[3];          /* coarsening factor towards the NEXT level (1 or 2) */
    long size;
    double *e[3];      /* face coefficients (+direction face of each node) */
    double *k2;
    double *u, *f, *r; /* correction, rhs, residual (levels >= 1; level 0 uses the caller's arrays) */
} semi_level;

static semi_level L[SEMI_MAX_LEVELS];
static int nlev = 0;
static int semi_sm = 3;

static inline long ID(const int *n, int i, int j, int k) { return ((long)i * n[1] + j) * n[2] + k; }
static inline int WR(int a, int n) { return a < 0 ? a + n : (a >= n ? a - n : a); }

/* ---------- generic kernels (coarse levels) ---------- */
static void op_residual(const semi_level *l, const double *u, const double *f, double *r) {
    const int *n = l->n;
    #pragma omp parallel for if(l->size >= 40000)
    for (int i = 0; i < n[0]; i++) {
        const int ip = WR(i + 1, n[0]), im = WR(i - 1, n[0]);
        for (int j = 0; j < n[1]; j++) {
            const int jp = WR(j + 1, n[1]), jm = WR(j - 1, n[1]);
            for (int k = 0; k < n[2]; k++) {
                const int kp = WR(k + 1, n[2]), km = WR(k - 1, n[2]);
                const long c = ID(n, i, j, k);
                const long xm = ID(n, im, j, k), ym = ID(n, i, jm, k), zm = ID(n, i, j, km);
                const double au = l->e[0][c] * (u[ID(n, ip, j, k)] - u[c]) + l->e[0][xm] * (u[xm] - u[c])
                                + l->e[1][c] * (u[ID(n, i, jp, k)] - u[c]) + l->e[1][ym] * (u[ym] - u[c])
                                + l->e[2][c] * (u[ID(n, i, j, kp)] - u[c]) + l->e[2][zm] * (u[zm] - u[c])
                                - l->k2[c] * u[c];
                r[c] = f[c] - au;
            }
        }
    }
}

static void rbgs(const semi_level *l, double *u, const double *f, int iters) {
    const int *n = l->n;
    for (int it = 0; it < iters; it++) {
        for (int color = 0; color < 2; color++) {
            #pragma omp parallel for if(l->size >= 40000)
            for (int i = 0; i < n[0]; i++) {
                const int ip = WR(i + 1, n[0]), im = WR(i - 1, n[0]);
                for (int j = 0; j < n[1]; j++) {
                    const int jp = WR(j + 1, n[1]), jm = WR(j - 1, n[1]);
                    for (int k = (i + j + color) % 2; k < n[2]; k += 2) {
                        const int kp = WR(k + 1, n[2]), km = WR(k - 1, n[2]);
                        const long c = ID(n, i, j, k);
                        const long xm = ID(n, im, j, k), ym = ID(n, i, jm, k), zm = ID(n, i, j, km);
                        double diag = l->e[0][c] + l->e[0][xm] + l->e[1][c] + l->e[1][ym] + l->e[2][c] + l->e[2][zm] + l->k2[c];
                        if (diag < 1e-14) diag = 1e-14;
                        const double s = l->e[0][c] * u[ID(n, ip, j, k)] + l->e[0][xm] * u[xm]
                                       + l->e[1][c] * u[ID(n, i, jp, k)] + l->e[1][ym] * u[ym]
                                       + l->e[2][c] * u[ID(n, i, j, kp)] + l->e[2][zm] * u[zm];
                        u[c] = (s - f[c]) / diag;
                    }
                }
            }
        }
    }
}

/* full-weighting weights for offset a in a coarsened direction (index a = -1,0,+1) */
static const double FW[3] = {0.25, 0.5, 0.25};

/* restrict a node field (residual / k2) */
static void restrict_node(const semi_level *fl, const semi_level *cl, const double *in, double *out) {
    const int *n = fl->n, *m = cl->n, *c = fl->c;
    #pragma omp parallel for if(cl->size >= 20000)
    for (int I = 0; I < m[0]; I++)
        for (int J = 0; J < m[1]; J++)
            for (int K = 0; K < m[2]; K++) {
                double s = 0.0;
                for (int a = (c[0] == 2 ? -1 : 0); a <= (c[0] == 2 ? 1 : 0); a++)
                    for (int b = (c[1] == 2 ? -1 : 0); b <= (c[1] == 2 ? 1 : 0); b++)
                        for (int d = (c[2] == 2 ? -1 : 0); d <= (c[2] == 2 ? 1 : 0); d++) {
                            const double w = (c[0] == 2 ? FW[a + 1] : 1.0) * (c[1] == 2 ? FW[b + 1] : 1.0) * (c[2] == 2 ? FW[d + 1] : 1.0);
                            s += w * in[ID(n, WR(c[0] * I + a, n[0]), WR(c[1] * J + b, n[1]), WR(c[2] * K + d, n[2]))];
                        }
                out[ID(m, I, J, K)] = s;
            }
}

/* coarse face coefficients along axis ax */
static void restrict_face(const semi_level *fl, semi_level *cl, int ax) {
    const int *n = fl->n, *m = cl->n, *c = fl->c;
    const double *in = fl->e[ax];
    double *out = cl->e[ax];
    const double s2 = (double)(c[ax] * c[ax]);
    #pragma omp parallel for if(cl->size >= 20000)
    for (int I = 0; I < m[0]; I++)
        for (int J = 0; J < m[1]; J++)
            for (int K = 0; K < m[2]; K++) {
                double s = 0.0;
                int lo[3], hi[3];
                for (int d = 0; d < 3; d++) {
                    lo[d] = (d != ax && c[d] == 2) ? -1 : 0;
                    hi[d] = (d != ax && c[d] == 2) ? 1 : 0;
                }
                for (int a = lo[0]; a <= hi[0]; a++)
                    for (int b = lo[1]; b <= hi[1]; b++)
                        for (int dd = lo[2]; dd <= hi[2]; dd++) {
                            const int off[3] = {a, b, dd};
                            double w = 1.0;
                            for (int d = 0; d < 3; d++) if (d != ax && c[d] == 2) w *= FW[off[d] + 1];
                            int p[3] = {c[0] * I + a, c[1] * J + b, c[2] * K + dd};
                            double v;
                            if (c[ax] == 2) {          /* two fine faces in series along ax */
                                int q[3] = {p[0], p[1], p[2]};
                                q[ax] += 1;
                                const double e1 = in[ID(n, WR(p[0], n[0]), WR(p[1], n[1]), WR(p[2], n[2]))];
                                const double e2 = in[ID(n, WR(q[0], n[0]), WR(q[1], n[1]), WR(q[2], n[2]))];
                                v = 2.0 * e1 * e2 / (e1 + e2 + 1e-300);
                            } else {
                                v = in[ID(n, WR(p[0], n[0]), WR(p[1], n[1]), WR(p[2], n[2]))];
                            }
                            s += w * v;
                        }
                out[ID(m, I, J, K)] = s / s2;
            }
}

/* u_fine += P u_coarse (linear interpolation in the coarsened directions) */
static void prolong_add(const semi_level *fl, const semi_level *cl, const double *uc, double *uf) {
    const int *n = fl->n, *m = cl->n, *c = fl->c;
    #pragma omp parallel for if(fl->size >= 40000)
    for (int i = 0; i < n[0]; i++)
        for (int j = 0; j < n[1]; j++)
            for (int k = 0; k < n[2]; k++) {
                const int p[3] = {i, j, k};
                int i0[3], i1[3]; double w1[3];
                for (int d = 0; d < 3; d++) {
                    if (c[d] == 2) {
                        i0[d] = p[d] / 2; w1[d] = (p[d] % 2) ? 0.5 : 0.0; i1[d] = WR(i0[d] + 1, m[d]);
                    } else { i0[d] = p[d]; i1[d] = p[d]; w1[d] = 0.0; }
                }
                double s = 0.0;
                for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int d = 0; d < 2; d++) {
                    const double w = (a ? w1[0] : 1.0 - w1[0]) * (b ? w1[1] : 1.0 - w1[1]) * (d ? w1[2] : 1.0 - w1[2]);
                    if (w == 0.0) continue;
                    s += w * uc[ID(m, a ? i1[0] : i0[0], b ? i1[1] : i0[1], d ? i1[2] : i0[2])];
                }
                uf[ID(n, i, j, k)] += s;
            }
}

/* coarsest level: CG on A u = f (A negative semi-definite; zero mode projected when k2 == 0) */
static void coarse_cg(semi_level *l, double *u, const double *f) {
    const long N = l->size;
    double *r = malloc(N * sizeof(double)), *p = malloc(N * sizeof(double)), *q = malloc(N * sizeof(double));
    double k2s = 0.0, fm = 0.0;
    for (long t = 0; t < N; t++) { k2s += l->k2[t]; fm += f[t]; }
    double *ff = malloc(N * sizeof(double));
    for (long t = 0; t < N; t++) ff[t] = f[t] - (k2s == 0.0 ? fm / N : 0.0);
    memset(u, 0, N * sizeof(double));
    memcpy(r, ff, N * sizeof(double));
    memcpy(p, r, N * sizeof(double));
    double rr = 0.0, r0;
    for (long t = 0; t < N; t++) rr += r[t] * r[t];
    r0 = rr;
    double *zero = calloc(N, sizeof(double));
    for (int it = 0; it < 300 && rr > 1e-12 * r0; it++) {
        op_residual(l, p, zero, q);              /* q = -A p */
        double pq = 0.0;
        for (long t = 0; t < N; t++) pq += p[t] * q[t];
        const double alpha = -rr / pq;           /* A p = -q */
        for (long t = 0; t < N; t++) { u[t] += alpha * p[t]; r[t] += alpha * q[t]; }
        double rn = 0.0;
        for (long t = 0; t < N; t++) rn += r[t] * r[t];
        for (long t = 0; t < N; t++) p[t] = r[t] + (rn / rr) * p[t];
        rr = rn;
    }
    free(r); free(p); free(q); free(ff); free(zero);
}

static void vcycle(int lv, double *u, double *f, int s1, int s2) {
    semi_level *l = &L[lv];
    if (lv == nlev - 1) { coarse_cg(l, u, f); return; }
    semi_level *cl = &L[lv + 1];
    if (lv == 0) {
        smooth_pb(f, u, s1, s2, (double)semi_sm, l->e[0], l->e[1], l->e[2], l->k2);
        laplace_filter_pb_residual(u, f, l->r, NULL, s1, s2, l->e[0], l->e[1], l->e[2], l->k2);
    } else {
        rbgs(l, u, f, semi_sm);
        op_residual(l, u, f, l->r);
    }
    restrict_node(l, cl, l->r, cl->f);
    memset(cl->u, 0, cl->size * sizeof(double));
    vcycle(lv + 1, cl->u, cl->f, 0, 0);
    prolong_add(l, cl, cl->u, u);
    if (lv == 0) smooth_pb(f, u, s1, s2, (double)semi_sm, l->e[0], l->e[1], l->e[2], l->k2);
    else rbgs(l, u, f, semi_sm);
}

void semi_mg_free(int s2);
static double mean_of(const double *a, long n) { double s = 0.0; for (long t = 0; t < n; t++) s += a[t]; return s / n; }

int semi_mg_setup(int s1, int s2, double *eps_x, double *eps_y, double *eps_z, double *k2) {
    const char *e = getenv("MAZE_MG_SEMI_SM");
    if (e != NULL && atoi(e) > 0) semi_sm = atoi(e);
    const int verbose = getenv("MAZE_MG_SEMI_VERBOSE") != NULL;
    semi_level *l0 = &L[0];
    l0->n[0] = s1; l0->n[1] = s2; l0->n[2] = s2; l0->size = (long)s1 * s2 * s2;
    l0->e[0] = eps_x; l0->e[1] = eps_y; l0->e[2] = eps_z; l0->k2 = k2;
    l0->r = mpi_grid_allocate(s1, s2);
    nlev = 1;
    while (nlev < SEMI_MAX_LEVELS) {
        semi_level *fl = &L[nlev - 1];
        if (fl->size <= SEMI_COARSEST_POINTS) break;
        double str[3], smax = 0.0;
        for (int d = 0; d < 3; d++) { str[d] = mean_of(fl->e[d], fl->size); if (str[d] > smax) smax = str[d]; }
        int any = 0;
        for (int d = 0; d < 3; d++) {
            fl->c[d] = (str[d] >= 0.5 * smax && fl->n[d] % 2 == 0 && fl->n[d] >= 4) ? 2 : 1;
            any |= (fl->c[d] == 2);
        }
        if (!any) break;
        semi_level *cl = &L[nlev];
        for (int d = 0; d < 3; d++) cl->n[d] = fl->n[d] / fl->c[d];
        cl->size = (long)cl->n[0] * cl->n[1] * cl->n[2];
        for (int d = 0; d < 3; d++) cl->e[d] = malloc(cl->size * sizeof(double));
        cl->k2 = malloc(cl->size * sizeof(double));
        cl->u = malloc(cl->size * sizeof(double)); cl->f = malloc(cl->size * sizeof(double)); cl->r = malloc(cl->size * sizeof(double));
        for (int d = 0; d < 3; d++) restrict_face(fl, cl, d);
        restrict_node(fl, cl, fl->k2, cl->k2);
        if (verbose) fprintf(stderr, "[semi-mg] level %d -> %d: %dx%dx%d -> %dx%dx%d (mean eps %.3g %.3g %.3g)\n",
                             nlev - 1, nlev, fl->n[0], fl->n[1], fl->n[2], cl->n[0], cl->n[1], cl->n[2], str[0], str[1], str[2]);
        nlev++;
    }
    /* too few levels (e.g. odd N): the coarsest CG would be as large as the fine grid -> caller falls back */
    if (nlev < 3 || L[nlev - 1].size > 8 * SEMI_COARSEST_POINTS) {
        static int warned = 0;
        if (!warned) { fprintf(stderr, "[semi-mg] grid %dx%dx%d cannot be semi-coarsened (coarsest %ld points): using the standard V-cycle\n",
                               s1, s2, s2, L[nlev - 1].size); warned = 1; }
        semi_mg_free(s2);
        return 0;
    }
    return nlev;
}

void semi_mg_free(int s2) {
    mpi_grid_free(L[0].r, s2);
    for (int lv = 1; lv < nlev; lv++) {
        for (int d = 0; d < 3; d++) free(L[lv].e[d]);
        free(L[lv].k2); free(L[lv].u); free(L[lv].f); free(L[lv].r);
    }
    nlev = 0;
}

/* one V-cycle on A u = f (u: in/out initial guess) */
void semi_mg_vcycle(double *f, double *u, int s1, int s2) { vcycle(0, u, f, s1, s2); }
