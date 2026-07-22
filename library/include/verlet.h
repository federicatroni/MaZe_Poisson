#ifndef VERLET_H
#define VERLET_H

#include "laplace.h"
#include "plcg.h"
#include "multigrid.h"
#include "multigrid_pb.h"

struct grid;

int verlet_poisson(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y, int size1, int size2,
    void (*precond)(double *, double *, int, int, int)
);
int verlet_poisson_multigrid(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y, int size1, int size2
);
int verlet_poisson_pb(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
);
int verlet_pb_multigrid(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y, 
    int size1, int size2, double *eps_x, double *eps_y, double *eps_z, double *k2_screen
);
int verlet_pb_multigrid_with_retry(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2, double *eps_x, double *eps_y, double *eps_z, double *k2_screen,
    int zero_mode
);
int verlet_pb_multigrid_eps_field(
    double tol, double h, double* phi, double* phi_prev, double* q, double* y,
    int size1, int size2, double *eps_x, double *eps_y, double *eps_z, double *k2_screen,
    struct grid *grid_ctx
);
#endif // VERLET_H
