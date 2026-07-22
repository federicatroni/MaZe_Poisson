#ifndef __LINALG_H
#define __LINALG_H

void dgetri(double *A, int n);
void dgemm(double *A, double *B, double *C, long int m, long int n, long int k);

void vec_copy(double *in, double *out, long int n);
void dscal(double *x, double alpha, long int n);
double ddot(double *u, double *v, long int n);
void daxpy(double *v, double *u, double alpha, long int n);
double norm(double *u, long int n);
double norm_inf(double *u, long int n);
double global_mean(const double *u, long int local_n, long int global_n);
void remove_global_mean(double *u, long int local_n, long int global_n);

#endif