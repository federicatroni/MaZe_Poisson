#ifndef __MP_MULTIGRID_SEMI_H
#define __MP_MULTIGRID_SEMI_H
/* [slit_dsl patch] semi-coarsening PB multigrid (see src/multigrid_semi.c); single MPI rank. */
int semi_mg_setup(int s1, int s2, double *eps_x, double *eps_y, double *eps_z, double *k2);  /* levels, 0 = fall back */
void semi_mg_vcycle(double *f, double *u, int s1, int s2);
void semi_mg_free(int s2);
#endif
