#include <math.h>
#include <stdlib.h>
#include "mpi_base.h"

#ifdef __cplusplus
#define EXTERN_C extern "C"                                                           
#else
#define EXTERN_C
#endif

#ifdef __LAPACK ///////////////////////////////////////////////////////////////////////////

#include <cblas.h>
#include <lapacke.h>

EXTERN_C void dgetri(double *A, int n) {
    int *ipiv = (int *)malloc(n * sizeof(int));

    lapack_int ret;

    ret = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A, n, ipiv);

    if (ret != 0) {
        mpi_fprintf(stderr, "dgetrf failed with info = %d\n", ret);
        exit(1);
    }

    ret = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A, n, ipiv);

    if (ret != 0) {
        mpi_fprintf(stderr, "dgetri failed with info = %d\n", ret);
        exit(1);
    }

    free(ipiv);
}

EXTERN_C double ddot(double *u, double *v, long int n) {
    double result = 0.0;
    result = cblas_ddot(n, u, 1, v, 1);
    allreduce_sum(&result, 1);
    return result;
}

EXTERN_C void daxpy(double *v, double *u, double alpha, long int n) {
    cblas_daxpy(n, alpha, v, 1, u, 1);
}

EXTERN_C void dgemm(double *A, double *B, double *C, long int m, long int n, long int k) {
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        m, n, k,
        // 1.0, A, m, B, k, 0.0, C, m
        1.0, A, k, B, n, 0.0, C, n
    );
}

EXTERN_C double norm(double *u, long int n) {
    return cblas_dnrm2(n, u, 1);
}

EXTERN_C double norm_inf(double *u, long int n) {
    double max_val = 0.0;
    
    // index of the maximum absolute value
    CBLAS_INDEX idx = cblas_idamax((int)n, u, 1);
    max_val = fabs(u[idx]);

    allreduce_max(&max_val, 1);
    return max_val;
}

/*
Scale a vector by a constant x = alpha * x
@param x: the vector to be scaled
@param alpha: the scaling constant
@param n: the size of the vector
*/
EXTERN_C void dscal(double *x, double alpha, long int n) {
    cblas_dscal(n, alpha, x, 1);
}

/*
Copy a vector from in to out
@param in: the input vector
@param out: the output vector
*/
EXTERN_C void vec_copy(double *in, double *out, long int n) {
    cblas_dcopy(n, in, 1, out, 1);
}

#else // __LAPACK ///////////////////////////////////////////////////////////////////////////

EXTERN_C void dgetri(double *A, int n) {
    mpi_fprintf(stderr, "dgerti is not implemented without LAPACK\n");
    exit(1);
}


/*
Copy a vector from in to out
@param in: the input vector
@param out: the output vector
*/
EXTERN_C void vec_copy(double *in, double *out, long int n) {
    long int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        out[i] = in[i];
    }
}

/*
Scale a vector by a constant x = alpha * x
@param x: the vector to be scaled
@param alpha: the scaling constant
@param n: the size of the vector
*/
EXTERN_C void dscal(double *x, double alpha, long int n) {
    long int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        x[i] *= alpha;
    }
}

/*
Compute the dot product of two vectors
@param u: the first vector
@param v: the second vector
@param n: the size of the vectors
@return the dot product of the two vectors
*/
EXTERN_C double ddot(double *u, double *v, long int n) {
    long int i;
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += u[i] * v[i];
    }
    allreduce_sum(&result, 1);
    return result;
}

/*
Compute the sum of two vectors scaled by a constant (u += alpha * v)
and store the result in the second vector
@param v: the first vector
@param u: the second vector
@param alpha: the scaling constant
@param n: the size of the vectors
*/
EXTERN_C void daxpy(double *v, double *u, double alpha, long int n) {
    long int i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        u[i] += alpha * v[i];
    }
}

/*
Compute the matrix-matrix product C = A * B
@param A: the first matrix (m x k)
@param B: the second matrix (k x n)
@param C: the output matrix (m x n)
@param m: the number of rows in A and C
@param n: the number of columns in B and C
@param k: the number of columns in A and rows in B
*/
EXTERN_C void dgemm(double *A, double *B, double *C, long int m, long int n, long int k) {
    long int i, j, l;
    #pragma omp parallel for private(i, j, l)
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
            for (l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
}

/*
Compute the Euclidean norm of a vector
@param u: the vector
@param n: the size of the vector
@return the Euclidean norm of the vector
*/
EXTERN_C double norm(double *u, long int n) {
    return sqrt(ddot(u, u, n));
}

/*
Compute the infinity norm (maximum absolute value) of a vector
@param u: the vector
@param n: the size of the vector
@return the infinity norm of the vector
*/
EXTERN_C double norm_inf(double *u, long int n) {
    long int i;
    double max_val = 0.0;
    double a; 

    #pragma omp parallel for private(a) reduction(max:max_val)
    for (i = 0; i < n; i++) {
        a = fabs(u[i]);
        if (a > max_val) max_val = a;
    }

    allreduce_max(&max_val, 1);
    return max_val;
}

#endif // __LAPACK ///////////////////////////////////////////////////////////////////////////

/*
Compute the mean of an MPI-distributed vector.  local_n is the number of
entries owned by this rank and global_n is the total number of entries.
*/
EXTERN_C double global_mean(const double *u, long int local_n, long int global_n) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long int i = 0; i < local_n; i++) {
        sum += u[i];
    }
    allreduce_sum(&sum, 1);

    return sum / (double)global_n;
}

/* Remove only the spatially constant component of an MPI-distributed vector. */
EXTERN_C void remove_global_mean(double *u, long int local_n, long int global_n) {
    const double mean = global_mean(u, local_n, global_n);

    #pragma omp parallel for
    for (long int i = 0; i < local_n; i++) {
        u[i] -= mean;
    }
}