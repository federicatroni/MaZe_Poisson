#include <stdio.h>
#include <math.h>

#include "mpi_base.h"


// CIC weight function as defined in the paper Im et al. (1998) - eqn 24
double spread_cic(double x, double L, double h) {
    x = fabs(x - round(x / L) * L);
    if (x >= h) {
        return 0.0;
    }
    return 1.0 - x / h;
}

double spread_spline_quadr(double x, double L, double h) {
    x = fabs(x - round(x / L) * L) / h;

    if (x < 0.5) {
        return 0.75 - x * x;
    } 
    if (x < 1.5) {
        return 0.5 * pow((1.5 - x), 2);
    }
    return 0.0;
}

double spread_spline_cubic(double x, double L, double h) {
    x = fabs(x - round(x / L) * L) / h;

    if (x < 1) {
        return (4.0 - 6.0 * pow(x, 2) + 3.0 * pow(x, 3)) / 6.0;
    } 
    if (x < 2) {
        return pow(2 - x, 3) / 6.0;
    }
    return 0.0;
}

#ifdef __MPI

double update_charges(
    int n_grid, int n_p, double h, int num_neigh,
    double *pos, long int *neighbors, double *charges, double *q,
    double (*g)(double, double, double)
) {
    int nn3 = num_neigh * 3;
    int n_loc = get_n_loc();
    int n_loc_start = get_n_start();

    long int ni, nj, nk, ni_loc;
    long int i1, i2;
    long int n2 = n_grid * n_grid;
    long int n3 = n_loc * n2;

    double px, py, pz;

    double L = n_grid * h;
    double q_tot = 0.0;
    double chg;
    double app, upd;

    #pragma omp parallel for
    for (long int i=0; i < n3; i++) {
        q[i] = 0.0;
    }

    #pragma omp parallel for private(i1, i2, ni, nj, nk, ni_loc, px, py, pz, app, upd, chg) reduction(+:q_tot)
    for (long int i=0; i<n_p; i++) {
        i1 = i * 3;
        i2 = i * nn3;

        chg = charges[i];
        px = pos[i1 + 0];
        py = pos[i1 + 1];
        pz = pos[i1 + 2];
        for (int j=0; j < nn3; j+=3) {
            ni = neighbors[i2 + j + 0];
            ni_loc = ni - n_loc_start;
            if (ni_loc < 0 || ni_loc >= n_loc) {
                continue;
            }
            nj = neighbors[i2 + j + 1];
            nk = neighbors[i2 + j + 2];
            app = g(px - ni*h, L, h) * g(py - nj*h, L, h) * g(pz - nk*h, L, h);
            upd = chg * app;
            q_tot += upd;
            #pragma omp atomic update
            q[ni_loc * n2 + nj * n_grid + nk] += upd;
        }
    }

    allreduce_sum(&q_tot, 1);

    return q_tot;
}

#else


double update_charges(
    int n_grid, int n_p, double h, int num_neigh,
    double *pos, long int *neighbors, double *charges, double *q,
    double (*g)(double, double, double)
) {
    int nn3 = num_neigh * 3;
    long int ni, nj, nk;
    long int i, i1, i2;
    long int n2 = n_grid * n_grid;
    long int n3 = n_grid * n2;

    double px, py, pz;

    double L = n_grid * h;
    double q_tot = 0.0;
    double chg;
    double app, upd;

    #pragma omp parallel for
    for (i=0; i < n3; i++) {
        q[i] = 0.0;
    }

    #pragma omp parallel for private(i1, i2, ni, nj, nk, px, py, pz, app, upd, chg) reduction(+:q_tot)
    for (i=0; i<n_p; i++) {
        i1 = i * 3;
        i2 = i * nn3;

        chg = charges[i];
        px = pos[i1 + 0];
        py = pos[i1 + 1];
        pz = pos[i1 + 2];
        for (int j=0; j < nn3; j+=3) {
            ni = neighbors[i2 + j + 0];
            nj = neighbors[i2 + j + 1];
            nk = neighbors[i2 + j + 2];
            app = g(px - ni*h, L, h) * g(py - nj*h, L, h) * g(pz - nk*h, L, h);
            upd = chg * app;
            q_tot += upd;
            #pragma omp atomic update
            q[ni * n2 + nj * n_grid + nk] += upd;
        }
    }

    return q_tot;
}

#endif