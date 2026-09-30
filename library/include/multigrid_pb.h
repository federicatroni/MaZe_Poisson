#ifndef __MP_MULTIGRID_PB_H
#define __MP_MULTIGRID_PB_H

#define MG_ITER_LIMIT_PB 1000

#define MG_SOLVE_SM_PB 4
#define MG_RECURSION_FACTOR_PB 2

/*
 * V-cycle settings.
 *  - MG_PB_CORR_SCALE: factor applied to the prolonged coarse-grid correction. The coarse operator is
 *    written in index units, so it is (h_c/h)^2 = 4 times weaker than the fine one and the restricted
 *    residual carries no compensating factor: the correction must be scaled by 4 (measured optimum 4-4.5).
 *  - MG_PB_MAXDEPTH: maximum number of coarsenings before the coarse CG (stops earlier when the grid gets small).
 * Both can be overridden with the environment variables MAZE_MG_CORR_SCALE and MAZE_MG_MAXDEPTH
 * (1.0 and 1 reproduce the original single-coarsening V-cycle).
 */
#define MG_PB_CORR_SCALE 4.0
#define MG_PB_MAXDEPTH 3
extern double mg_pb_corr_scale;
extern int mg_pb_maxdepth;
void mg_pb_env_init(void);

void restriction_eps(double *eps_in, double *eps_out, int s1, int s2, int axis);
void restriction_k2screen(const double *in, double *out, int s1, int s2, int n_start);
void smooth_pb(double *in, double *out, int s1, int s2, double tol, double *eps_x, double *eps_y, double *eps_z, double *k2_screen);

int multigrid_pb_apply(
    double *in, double *out, int s1, int s2, int n_start1, int sm, double *eps_x, double *eps_y, double *eps_z, double *k2_screen
);


#endif // __MP_MULTIGRID_PB_H