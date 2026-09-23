#ifndef __LAPLACE_H
#define __LAPLACE_H

void laplace_filter(double *u, double *u_new, int size1, int size2);
void laplace_filter_pb(
    double *u, double *u_new, int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
);

void laplace_filter_pb_residual(
    double *u, double *rhs, double *out, double *norm, int size1, int size2,
    double *eps_x, double *eps_y, double *eps_z, double *k2_screen
);

#endif // __LAPLACE_H
