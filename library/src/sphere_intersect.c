#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sphere_intersect.h"

/* ------------------------------------------------------------------ */
/* Helper    // Vincenzo Di Florio, 04.2026                                                           */
/* ------------------------------------------------------------------ */

static inline double min_image_1d(double d, double L)
{
    if (L <= 0.0) return d;
    return d - L * nearbyint(d / L);
}

/*
 * Solve the quadratic equation for the intersection between a parametric ray
 * P(t) = start + t * (edge_dir * h) and a sphere with center c and radius r.
 *
 * dx, dy, dz: components of (start - c), already minimum-image adjusted.
 * edge_dir: 0=X, 1=Y, 2=Z.
 * h: edge length.
 *
 * Writes unclamped values to t1_out and t2_out, with t1 <= t2.
 * Returns 1 if an intersection exists, 0 if the discriminant is negative.
 */
static int solve_sphere_edge(double dx, double dy, double dz,
                              double r, double h, int edge_dir,
                              double *t1_out, double *t2_out)
{
    /* A t^2 + B t + C = 0
     * A = h^2, always positive
     * B = 2 * d_along * h, where d_along is the component of (start-c) along the edge
     * C = |start - c|^2 - r^2
     */
    double d_along = (edge_dir == 0) ? dx : (edge_dir == 1) ? dy : dz;
    double A = h * h;
    double B = 2.0 * d_along * h;
    double C = dx*dx + dy*dy + dz*dz - r*r;

    double disc = B*B - 4.0*A*C;
    if (disc < 0.0) return 0;

    double sqrtd = sqrt(disc);
    double inv2A = 0.5 / A;
    *t1_out = (-B - sqrtd) * inv2A;
    *t2_out = (-B + sqrtd) * inv2A;
    return 1;
}

/* ------------------------------------------------------------------ */
/* is_in_molecule_sphere                                                */
/* ------------------------------------------------------------------ */

int is_in_molecule_sphere(const particles *p, double x, double y, double z, double L)
{
    for (int ip = 0; ip < p->n_p; ip++) {
        double dx = min_image_1d(x - p->pos[ip * 3 + 0], L);
        double dy = min_image_1d(y - p->pos[ip * 3 + 1], L);
        double dz = min_image_1d(z - p->pos[ip * 3 + 2], L);
        double r  = p->solv_radii[ip];
        if (dx*dx + dy*dy + dz*dz <= r*r) return 1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* sphere_edge_fraction                                                 */
/* ------------------------------------------------------------------ */

/* Maximum interval capacity, one per atom. Increase if N_atoms > 128. */
#define MAX_SPHERE_ATOMS 256

/* Total length of the union of n >= 2 intervals [lo,hi] (sorted in place). */
static double merge_interval_fraction(double *lo, double *hi, int n_intervals)
{
    /* Sort by lo using insertion sort; n is small. */
    for (int i = 1; i < n_intervals; i++) {
        double lv = lo[i], hv = hi[i];
        int j = i - 1;
        while (j >= 0 && lo[j] > lv) {
            lo[j+1] = lo[j];
            hi[j+1] = hi[j];
            j--;
        }
        lo[j+1] = lv;
        hi[j+1] = hv;
    }

    /* Merge intervals and sum their lengths. */
    double frac    = 0.0;
    double cur_lo  = lo[0], cur_hi = hi[0];
    for (int i = 1; i < n_intervals; i++) {
        if (lo[i] <= cur_hi) {
            if (hi[i] > cur_hi) cur_hi = hi[i];
        } else {
            frac  += cur_hi - cur_lo;
            cur_lo = lo[i];
            cur_hi = hi[i];
        }
    }
    frac += cur_hi - cur_lo;

    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
    return frac;
}

double sphere_edge_fraction(const particles *p,
    double x0, double y0, double z0,
    double h, int dir, double L)
{
    double lo[MAX_SPHERE_ATOMS], hi[MAX_SPHERE_ATOMS];
    int n_intervals = 0;

    for (int ip = 0; ip < p->n_p && n_intervals < MAX_SPHERE_ATOMS; ip++) {
        double dx = min_image_1d(x0 - p->pos[ip * 3 + 0], L);
        double dy = min_image_1d(y0 - p->pos[ip * 3 + 1], L);
        double dz = min_image_1d(z0 - p->pos[ip * 3 + 2], L);
        double r  = p->solv_radii[ip];

        double t1, t2;
        if (!solve_sphere_edge(dx, dy, dz, r, h, dir, &t1, &t2)) continue;

        /* Clamp to [0,1]. */
        if (t1 < 0.0) t1 = 0.0;
        if (t2 > 1.0) t2 = 1.0;
        if (t1 >= t2) continue;

        lo[n_intervals] = t1;
        hi[n_intervals] = t2;
        n_intervals++;
    }

    if (n_intervals == 0) return 0.0;
    if (n_intervals == 1) return hi[0] - lo[0];
    return merge_interval_fraction(lo, hi, n_intervals);
}

/*
 * Same as calling sphere_edge_fraction for dir = 0, 1, 2, but shares the minimum
 * image work and skips spheres that provably cannot cross an edge (margin 1e-6
 * on the rejection tests, so the surviving intervals are exactly the same).
 */
/* Add the clamped chord of sphere (radius r, centre offset d) on the three edges from one node. */
static inline void edge_add_sphere(const double *d, double r, double h,
    double lo[3][MAX_SPHERE_ATOMS], double hi[3][MAX_SPHERE_ATOMS], int *n_int)
{
    const double margin = 1.000001;
    const double r2m = r * r * margin;
    const double rm  = r * margin;
    const double rhm = (r + h) * margin;

    for (int dir = 0; dir < 3; dir++) {
        double along = d[dir];
        double perp2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2] - along*along;
        if (perp2 > r2m || along > rm || along < -rhm) continue;

        double t1, t2;
        if (!solve_sphere_edge(d[0], d[1], d[2], r, h, dir, &t1, &t2)) continue;
        if (t1 < 0.0) t1 = 0.0;
        if (t2 > 1.0) t2 = 1.0;
        if (t1 >= t2) continue;
        lo[dir][n_int[dir]] = t1;
        hi[dir][n_int[dir]] = t2;
        n_int[dir]++;
    }
}

static inline void edge_finish(double lo[3][MAX_SPHERE_ATOMS], double hi[3][MAX_SPHERE_ATOMS],
    const int *n_int, double *frac)
{
    for (int dir = 0; dir < 3; dir++) {
        if (n_int[dir] == 0)      frac[dir] = 0.0;
        else if (n_int[dir] == 1) frac[dir] = hi[dir][0] - lo[dir][0];
        else                      frac[dir] = merge_interval_fraction(lo[dir], hi[dir], n_int[dir]);
    }
}

void sphere_edge_fractions3(const particles *p,
    double x0, double y0, double z0,
    double h, double L, double *frac)
{
    double lo[3][MAX_SPHERE_ATOMS], hi[3][MAX_SPHERE_ATOMS];
    int n_int[3] = {0, 0, 0};

    for (int ip = 0; ip < p->n_p && ip < MAX_SPHERE_ATOMS; ip++) {
        double d[3];
        d[0] = min_image_1d(x0 - p->pos[ip * 3 + 0], L);
        d[1] = min_image_1d(y0 - p->pos[ip * 3 + 1], L);
        d[2] = min_image_1d(z0 - p->pos[ip * 3 + 2], L);
        edge_add_sphere(d, p->solv_radii[ip], h, lo, hi, n_int);
    }
    edge_finish(lo, hi, n_int, frac);
}

/*
 * Evaluate a whole grid line (fixed x, y; z = k*h for k = 0..n-1) given the list of spheres that
 * can reach it (cand[0..nc), ascending particle index, see sphere_build_line_candidates):
 * inside[k] = is_in_molecule_sphere(x, y, z) and frac[dir][k] = sphere_edge_fraction(dir).
 * Spheres left out of the list cannot touch the line (bounding test with margin 1e-6), and the
 * others are processed in the original order, so the results equal the per-node functions.
 * Requires nc <= MAX_SPHERE_ATOMS.
 */
void sphere_line_eval_cand(const particles *p, double x, double y, double h, double L, int n,
    const int *cand, int nc,
    unsigned int *inside, double *frac_x, double *frac_y, double *frac_z)
{
    double cdx[MAX_SPHERE_ATOMS], cdy[MAX_SPHERE_ATOMS];
    for (int c = 0; c < nc; c++) {
        cdx[c] = min_image_1d(x - p->pos[cand[c] * 3 + 0], L);
        cdy[c] = min_image_1d(y - p->pos[cand[c] * 3 + 1], L);
    }

    for (int k = 0; k < n; k++) {
        const double z = k * h;
        double lo[3][MAX_SPHERE_ATOMS], hi[3][MAX_SPHERE_ATOMS];
        int n_int[3] = {0, 0, 0};
        unsigned int in = 0u;
        for (int c = 0; c < nc; c++) {
            const int ip = cand[c];
            double d[3];
            d[0] = cdx[c];
            d[1] = cdy[c];
            d[2] = min_image_1d(z - p->pos[ip * 3 + 2], L);
            const double r = p->solv_radii[ip];
            if (d[0]*d[0] + d[1]*d[1] + d[2]*d[2] <= r*r) in = 1u;
            edge_add_sphere(d, r, h, lo, hi, n_int);
        }
        double fr[3];
        edge_finish(lo, hi, n_int, fr);
        inside[k] = in;
        frac_x[k] = fr[0];
        frac_y[k] = fr[1];
        frac_z[k] = fr[2];
    }
}

/*
 * Build, in O(n_particles + n_lines), the list of spheres that can reach each grid line
 * (line id = i_local * n + j, x = (i_local + n_start) * h, y = j * h), in CSR form:
 * the candidates of line l are cand[offsets[l] .. offsets[l+1]), in ascending particle index.
 * Each particle only visits the lines inside its bounding box (plus one line of slack), and
 * the same bounding test as before (margin 1e-6, computed with the same min_image values)
 * decides the membership. Returns 0 on success, -1 if a line has more than MAX_SPHERE_ATOMS
 * candidates (the caller must then use the per-node functions). offsets has n_local*n + 1
 * entries; both arrays are malloc'd and must be freed by the caller.
 */
int sphere_build_line_candidates(const particles *p, int n, int n_local, int n_start,
    double h, double L, int **offsets_out, int **cand_out)
{
    const double margin = 1.000001;
    const long n_lines = (long)n_local * n;
    int *offsets = (int *)calloc((size_t)n_lines + 1, sizeof(int));
    int *fillpos = (int *)malloc((size_t)(n_lines + 1) * sizeof(int));
    int *gi = (int *)malloc((size_t)n * sizeof(int));
    int *gj = (int *)malloc((size_t)n * sizeof(int));
    double *dxv = (double *)malloc((size_t)n * sizeof(double));
    double *dyv = (double *)malloc((size_t)n * sizeof(double));
    int *cand = NULL;
    int status = 0;

    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            long total = 0;
            for (long l = 0; l < n_lines; l++) {
                int cnt = offsets[l];
                offsets[l] = (int)total;
                total += cnt;
            }
            offsets[n_lines] = (int)total;
            cand = (int *)malloc((size_t)(total > 0 ? total : 1) * sizeof(int));
            for (long l = 0; l <= n_lines; l++) fillpos[l] = offsets[l];
        }

        for (int ip = 0; ip < p->n_p; ip++) {
            const double r = p->solv_radii[ip];
            const double px = p->pos[ip * 3 + 0], py = p->pos[ip * 3 + 1];
            const int span = (int)ceil((r + h) * margin / h) + 1;

            int ni, nj;
            if (2 * span + 1 >= n) {           /* box wider than the grid: visit everything */
                ni = nj = n;
                for (int t = 0; t < n; t++) { gi[t] = t; gj[t] = t; }
            } else {
                const int iq = (int)floor(px / h), jq = (int)floor(py / h);
                ni = nj = 2 * span + 1;
                for (int t = 0; t < ni; t++) {
                    gi[t] = (((iq - span + t) % n) + n) % n;
                    gj[t] = (((jq - span + t) % n) + n) % n;
                }
            }
            /* per-plane / per-row displacements, computed exactly as the per-node code does */
            for (int a = 0; a < ni; a++) dxv[a] = min_image_1d(gi[a] * h - px, L);
            for (int b = 0; b < nj; b++) dyv[b] = min_image_1d(gj[b] * h - py, L);

            for (int a = 0; a < ni; a++) {
                const int il = gi[a] - n_start;
                if (il < 0 || il >= n_local) continue;
                const double dx = dxv[a];
                if (dx > r * margin || dx < -(r + h) * margin) continue;
                for (int b = 0; b < nj; b++) {
                    const double dy = dyv[b];
                    if (dy > r * margin || dy < -(r + h) * margin) continue;
                    const long l = (long)il * n + gj[b];
                    if (pass == 0) offsets[l]++;
                    else cand[fillpos[l]++] = ip;
                }
            }
        }
    }

    for (long l = 0; l < n_lines; l++)
        if (offsets[l + 1] - offsets[l] > MAX_SPHERE_ATOMS) { status = -1; break; }

    free(fillpos); free(gi); free(gj); free(dxv); free(dyv);
    *offsets_out = offsets;
    *cand_out = cand;
    return status;
}

/* ------------------------------------------------------------------ */
/* sphere_edge_surface_inters                                           */
/* ------------------------------------------------------------------ */

/* Event: entering (+1) or leaving (-1) a sphere along the edge. */
typedef struct { double t; int atom; int sign; } SphereEvent;

#define MAX_EVENTS (2 * MAX_SPHERE_ATOMS)

int sphere_edge_surface_inters(
    const particles *p,
    double x0, double y0, double z0,
    double h, int dir, double L,
    double *t_out, double *nx_out, double *ny_out, double *nz_out,
    int max_inters)
{
    SphereEvent events[MAX_EVENTS];
    int n_events     = 0;
    int initial_count = 0; /* Number of spheres containing the start point (t=0). */

    for (int ip = 0; ip < p->n_p && n_events + 2 <= MAX_EVENTS; ip++) {
        double dx = min_image_1d(x0 - p->pos[ip * 3 + 0], L);
        double dy = min_image_1d(y0 - p->pos[ip * 3 + 1], L);
        double dz = min_image_1d(z0 - p->pos[ip * 3 + 2], L);
        double r  = p->solv_radii[ip];

        double t1, t2;
        if (!solve_sphere_edge(dx, dy, dz, r, h, dir, &t1, &t2)) continue;

        /* Sphere is beyond the edge end or before the edge start. */
        if (t2 <= 0.0) continue;
        if (t1 >= 1.0) continue;

        int starts_inside = (t1 < 0.0);
        int ends_inside   = (t2 > 1.0);

        if (starts_inside) {
            /* The edge starts inside this sphere. */
            initial_count++;
            if (!ends_inside) {
                /* Leaves the sphere at t2, inside [0,1]. */
                events[n_events].t    = t2;
                events[n_events].atom = ip;
                events[n_events].sign = -1;
                n_events++;
            }
        } else {
            /* Enters the sphere at t1, inside [0,1]. */
            events[n_events].t    = t1;
            events[n_events].atom = ip;
            events[n_events].sign = +1;
            n_events++;
            if (!ends_inside) {
                /* Leaves the sphere at t2, inside [0,1]. */
                events[n_events].t    = t2;
                events[n_events].atom = ip;
                events[n_events].sign = -1;
                n_events++;
            }
        }
    }

    if (n_events == 0) return 0;

    /* Sort events by t using insertion sort. */
    for (int i = 1; i < n_events; i++) {
        SphereEvent ev = events[i];
        int j = i - 1;
        while (j >= 0 && events[j].t > ev.t) {
            events[j+1] = events[j];
            j--;
        }
        events[j+1] = ev;
    }

    /* Sweep to find 0 <-> positive transitions, i.e. the external surface. */
    int count   = initial_count;
    int n_found = 0;

    for (int e = 0; e < n_events && n_found < max_inters; e++) {
        int prev_count = count;
        count += events[e].sign;

        /* External surface: transition between inside (count>0) and outside (count==0). */
        if ((prev_count == 0) != (count == 0)) {
            int    ip = events[e].atom;
            double t  = events[e].t;

            /* 3D position of the intersection point. */
            double px = x0, py = y0, pz = z0;
            if      (dir == 0) px += t * h;
            else if (dir == 1) py += t * h;
            else               pz += t * h;

            /* Outward sphere normal: unit vector (P - center) / r. */
            double cx = p->pos[ip * 3 + 0];
            double cy = p->pos[ip * 3 + 1];
            double cz = p->pos[ip * 3 + 2];
            double r  = p->solv_radii[ip];

            double npx = min_image_1d(px - cx, L);
            double npy = min_image_1d(py - cy, L);
            double npz = min_image_1d(pz - cz, L);
            double inv_r = 1.0 / r;

            t_out [n_found] = t;
            nx_out[n_found] = npx * inv_r;
            ny_out[n_found] = npy * inv_r;
            nz_out[n_found] = npz * inv_r;
            n_found++;
        }
    }

    return n_found;
}

#undef MAX_EVENTS
#undef MAX_SPHERE_ATOMS
