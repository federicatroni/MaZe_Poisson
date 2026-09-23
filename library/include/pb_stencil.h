#ifndef __PB_STENCIL_H
#define __PB_STENCIL_H

/*
 * One grid row (fixed i, j; k = 0..s2-1, periodic in k) of the PB operator
 *   out[k] = sum_faces eps_face * u_neighbour - u_centre * (sum_faces eps_face + k2)
 * c  : offset of the row (i0 + j0), xp / xm : same row in plane i+1 / i-1,
 * yp / ym : row j+1 / j-1 (periodic) in plane i.
 * The interior of the row has affine indices, so it vectorises; the two periodic
 * ends in k are handled separately. Evaluation order matches laplace_filter_pb.
 */
static inline void pb_stencil_row(
    const double *restrict u, const double *restrict ex, const double *restrict ey,
    const double *restrict ez, const double *restrict k2,
    long c, long xp, long xm, long yp, long ym, int s2, double *restrict out
) {
    for (int k = 1; k < s2 - 1; k++) {
        out[k] = (
            u[xp + k]     * ex[c + k] +
            u[xm + k]     * ex[xm + k] +
            u[yp + k]     * ey[c + k] +
            u[ym + k]     * ey[ym + k] +
            u[c + k + 1]  * ez[c + k] +
            u[c + k - 1]  * ez[c + k - 1] -
            u[c + k] * (
                ex[c + k] + ex[xm + k] +
                ey[c + k] + ey[ym + k] +
                ez[c + k] + ez[c + k - 1] +
                k2[c + k]
            )
        );
    }
    for (int e = 0; e < 2; e++) {
        const int k  = e ? s2 - 1 : 0;
        const int kp = e ? 0 : 1;
        const int km = e ? s2 - 2 : s2 - 1;
        out[k] = (
            u[xp + k]     * ex[c + k] +
            u[xm + k]     * ex[xm + k] +
            u[yp + k]     * ey[c + k] +
            u[ym + k]     * ey[ym + k] +
            u[c + kp]     * ez[c + k] +
            u[c + km]     * ez[c + km] -
            u[c + k] * (
                ex[c + k] + ex[xm + k] +
                ey[c + k] + ey[ym + k] +
                ez[c + k] + ez[c + km] +
                k2[c + k]
            )
        );
    }
}

#endif // __PB_STENCIL_H
