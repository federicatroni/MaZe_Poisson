#ifndef __MP_MULTIGRID_H
#define __MP_MULTIGRID_H

#define MG_ITER_LIMIT 1000

#define MG_SOLVE_SM 3
#define MG_RECURSION_FACTOR 2

/*
 * V-cycle settings of the plain Poisson multigrid (same reasoning as the PB one, see multigrid_pb.h):
 * the coarse stencil is written in index units, so the prolonged coarse correction must be scaled by
 * (h_c/h)^2 = 4, and more levels (up to MG_MAXDEPTH coarsenings before the coarse CG) then become effective.
 * Overridable with MAZE_MG_CORR_SCALE / MAZE_MG_MAXDEPTH (1.0 and 1 reproduce the original cycle).
 * MAZE_MG_KEEP_Y=1 keeps the previous y as initial guess (default: start from y = 0).
 */
#define MG_CORR_SCALE 4.0
#define MG_MAXDEPTH 3
extern double mg_corr_scale;
extern int mg_maxdepth;
extern int mg_keep_y;
void mg_env_init(void);

void prolong(double *in, double *out, int s1, int s2, int ts1, int ts2, int tns);
void restriction(double *in, double *out, int s1, int s2, int n_start);
void smooth(double *in, double *out, int s1, int s2, double tol);

int multigrid_apply(
    double *in, double *out, int s1, int s2, int n_start1, int sm
);


#endif // __MP_MULTIGRID_H