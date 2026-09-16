#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi_base.h"
#include "mp_structs.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * A compact row of the radial stencil.  Grouping coefficients by (dx,dy)
 * leaves dz contiguous in both the coefficient array and, except at the
 * periodic seam, the charge grid.  This is substantially cheaper than an
 * array of independent (dx,dy,dz) tuples in the innermost loop.
 */
typedef struct {
    int dx;
    int dy;
    int dz_first;
    int dz_count;
    long weight_first;
} wendland_row;

typedef struct {
    int n;
    int n_local;
    int halo;
    int disp_min;
    int disp_max;
    int n_rows;
    wendland_row *rows;
    int *dx_row_first;
    double *weights;
    double *scatter_weights;
    double *extended;
    double *output;
    long *nonzero_indices;
    int thread_capacity;
    int *thread_output_begin;
    int *thread_output_end;
    long *thread_nz_begin;
    long *thread_nz_end;
    long *thread_buffer_capacity;
    double **thread_buffers;
    long *source_work;
} wendland_nofft_kernel;

static int max_threads(void) {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

static int thread_number(void) {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

static int periodic_index(int i, int n) {
    i %= n;
    return i < 0 ? i + n : i;
}

static double wendland_value(int order, double t) {
    const double one_minus_t = 1.0 - t;
    const double omt2 = one_minus_t * one_minus_t;
    const double omt4 = omt2 * omt2;
    if (order == 2) {
        return omt4 * (4.0 * t + 1.0);
    }
    return omt4 * omt2 * (35.0 * t * t + 18.0 * t + 3.0);
}

#ifdef __MPI
static int owner_of_plane(int global_i, const mpi_data *mpid) {
    for (int rank = 0; rank < mpid->size; ++rank) {
        int start = mpid->n_start_list[rank];
        int count = mpid->n_loc_list[rank];
        if (count > 0 && global_i >= start && global_i < start + count) return rank;
    }
    return -1;  /* The decomposition must cover every global x plane. */
}
#endif

/*
 * Fill an x-extended copy of q.  Only halo planes cross MPI ranks.  Unlike
 * repeated nearest-neighbour exchanges, this remains correct when a rank
 * owns fewer x planes than the kernel radius.  Tags are the destination's
 * extended-plane indices, so multiple planes from one peer are unambiguous.
 */
static void fill_extended_grid(grid *grid, wendland_nofft_kernel *kernel) {
    const int n = kernel->n;
    const int n_local = kernel->n_local;
    const int halo = kernel->halo;
    const int ext_planes = n_local + 2 * halo;
    const long n2 = (long)n * n;
    mpi_data *mpid = get_mpi_data();

    /* FFTW decompositions may leave trailing ranks without x planes. */
    if (n_local == 0) return;

    if (mpid->size == 1) {
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < ext_planes; ++e) {
            int source = periodic_index(e - halo, n);
            memcpy(kernel->extended + (long)e * n2, grid->q + (long)source * n2,
                   n2 * sizeof(double));
        }
        return;
    }

#ifdef __MPI
    const int rank = mpid->rank;
    const int max_requests = ext_planes + n + 2 * halo * mpid->size;
    MPI_Request *requests = malloc((long)max_requests * sizeof(*requests));
    if (requests == NULL) {
        mpi_fprintf(stderr, "Unable to allocate Wendland halo MPI requests\n");
        exit(EXIT_FAILURE);
    }
    int request_count = 0;

    for (int e = 0; e < ext_planes; ++e) {
        int global_i = periodic_index(grid->n_start + e - halo, n);
        int owner = owner_of_plane(global_i, mpid);
        if (owner == rank) {
            int source = global_i - grid->n_start;
            memcpy(kernel->extended + (long)e * n2, grid->q + (long)source * n2,
                   n2 * sizeof(double));
        } else {
            MPI_Irecv(kernel->extended + (long)e * n2, n2, MPI_DOUBLE, owner, e,
                      mpid->comm, &requests[request_count++]);
        }
    }

    for (int target = 0; target < mpid->size; ++target) {
        int target_local = mpid->n_loc_list[target];
        if (target == rank || target_local == 0) continue;
        int target_ext_planes = target_local + 2 * halo;
        for (int e = 0; e < target_ext_planes; ++e) {
            int global_i = periodic_index(mpid->n_start_list[target] + e - halo, n);
            if (owner_of_plane(global_i, mpid) == rank) {
                int source = global_i - grid->n_start;
                MPI_Isend(grid->q + (long)source * n2, n2, MPI_DOUBLE, target, e,
                          mpid->comm, &requests[request_count++]);
            }
        }
    }

    MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);
    free(requests);
#endif
}

void smooth_charges_wendland_nofft_init(grid *grid, int order) {
    const int n = grid->n;
    const int n_local = grid->n_local;
    const double sigma = grid->smoothing_sigma / grid->h;
    const double sigma2 = sigma * sigma;
    /* Same canonical minimum-image convention used by the FFT kernel. */
    const int canonical_min = -(n - 1) / 2;
    const int canonical_max = n / 2;
    const int support = (int)ceil(sigma);
    const int disp_min = -support > canonical_min ? -support : canonical_min;
    const int disp_max = support < canonical_max ? support : canonical_max;
    const int halo = (-disp_min > disp_max) ? -disp_min : disp_max;

    int row_count = 0;
    long weight_count = 0;
    for (int dx = disp_min; dx <= disp_max; ++dx) {
        for (int dy = disp_min; dy <= disp_max; ++dy) {
            int count = 0;
            /* Descending dz makes q[k-dz] advance contiguously in memory. */
            for (int dz = disp_max; dz >= disp_min; --dz) {
                double r2 = (double)dx * dx + (double)dy * dy + (double)dz * dz;
                if (r2 <= sigma2) ++count;
            }
            if (count > 0) {
                ++row_count;
                weight_count += count;
            }
        }
    }

    wendland_nofft_kernel *kernel = calloc(1, sizeof(*kernel));
    if (kernel == NULL) {
        mpi_fprintf(stderr, "Unable to allocate no-FFT Wendland kernel\n");
        exit(EXIT_FAILURE);
    }
    kernel->n = n;
    kernel->n_local = n_local;
    kernel->halo = halo;
    kernel->disp_min = disp_min;
    kernel->disp_max = disp_max;
    kernel->n_rows = row_count;
    kernel->rows = malloc((long)row_count * sizeof(*kernel->rows));
    kernel->dx_row_first = malloc(
        (long)(disp_max - disp_min + 2) * sizeof(*kernel->dx_row_first));
    kernel->weights = malloc(weight_count * sizeof(*kernel->weights));
    kernel->scatter_weights = malloc(weight_count * sizeof(*kernel->scatter_weights));
    kernel->extended = malloc((long)(n_local + 2 * kernel->halo) * n * n * sizeof(double));
    kernel->nonzero_indices = malloc((long)(n_local + 2 * kernel->halo) * n * n * sizeof(long));
    long output_size = (long)n_local * n * n;
    kernel->output = malloc((output_size > 0 ? output_size : 1) * sizeof(double));
    kernel->thread_capacity = max_threads();
    kernel->thread_output_begin = malloc((long)kernel->thread_capacity * sizeof(int));
    kernel->thread_output_end = malloc((long)kernel->thread_capacity * sizeof(int));
    kernel->thread_nz_begin = malloc((long)kernel->thread_capacity * sizeof(long));
    kernel->thread_nz_end = malloc((long)kernel->thread_capacity * sizeof(long));
    kernel->thread_buffer_capacity = calloc(
        (long)kernel->thread_capacity, sizeof(*kernel->thread_buffer_capacity));
    kernel->thread_buffers = calloc(
        (long)kernel->thread_capacity, sizeof(*kernel->thread_buffers));
    kernel->source_work = malloc(
        (long)(n_local + 2 * kernel->halo) * sizeof(*kernel->source_work));
    if (kernel->rows == NULL || kernel->dx_row_first == NULL ||
        kernel->weights == NULL || kernel->scatter_weights == NULL ||
        kernel->extended == NULL || kernel->output == NULL ||
        kernel->nonzero_indices == NULL || kernel->source_work == NULL) {
        mpi_fprintf(stderr, "Unable to allocate no-FFT Wendland stencil storage\n");
        exit(EXIT_FAILURE);
    }
    if (kernel->thread_output_begin == NULL || kernel->thread_output_end == NULL ||
        kernel->thread_nz_begin == NULL || kernel->thread_nz_end == NULL ||
        kernel->thread_buffer_capacity == NULL || kernel->thread_buffers == NULL) {
        mpi_fprintf(stderr, "Unable to allocate no-FFT Wendland thread metadata\n");
        exit(EXIT_FAILURE);
    }

    int row = 0;
    long weight = 0;
    double sum = 0.0;
    for (int dx = disp_min; dx <= disp_max; ++dx) {
        kernel->dx_row_first[dx - disp_min] = row;
        for (int dy = disp_min; dy <= disp_max; ++dy) {
            int dz_first = 0;
            int dz_count = 0;
            long first_weight = weight;
            /* Descending dz makes q[k-dz] advance contiguously in memory. */
            for (int dz = disp_max; dz >= disp_min; --dz) {
                double r2 = (double)dx * dx + (double)dy * dy + (double)dz * dz;
                if (r2 <= sigma2) {
                    if (dz_count == 0) dz_first = dz;
                    double value = wendland_value(order, sqrt(r2) / sigma);
                    kernel->weights[weight++] = value;
                    sum += value;
                    ++dz_count;
                }
            }
            if (dz_count > 0) {
                kernel->rows[row].dx = dx;
                kernel->rows[row].dy = dy;
                kernel->rows[row].dz_first = dz_first;
                kernel->rows[row].dz_count = dz_count;
                kernel->rows[row].weight_first = first_weight;
                ++row;
            }
        }
    }
    kernel->dx_row_first[disp_max - disp_min + 1] = row;
    for (long w = 0; w < weight_count; ++w) kernel->weights[w] /= sum;
    for (int r = 0; r < row_count; ++r) {
        const wendland_row *stencil_row = kernel->rows + r;
        const long first = stencil_row->weight_first;
        const int count = stencil_row->dz_count;
        for (int z = 0; z < count; ++z) {
            kernel->scatter_weights[first + z] = kernel->weights[first + count - 1 - z];
        }
    }
    const int ext_planes = n_local + 2 * halo;
    for (int source_i = 0; source_i < ext_planes; ++source_i) {
        int first_dx = halo - source_i;
        int last_dx = halo - source_i + n_local - 1;
        if (first_dx < disp_min) first_dx = disp_min;
        if (last_dx > disp_max) last_dx = disp_max;
        long work = 0;
        for (int dx = first_dx; dx <= last_dx; ++dx) {
            const int r_begin = kernel->dx_row_first[dx - disp_min];
            const int r_end = kernel->dx_row_first[dx - disp_min + 1];
            for (int r = r_begin; r < r_end; ++r) work += kernel->rows[r].dz_count;
        }
        kernel->source_work[source_i] = work;
    }

    grid->smoothing_kernel = kernel;

    if (getenv("MAZE_WENDLAND_DEBUG") != NULL) {
        mpi_fprintf(stderr,
            "[wendland-debug] init: n=%d n_local=%d sigma_grid=%.6f support=%d "
            "disp=[%d,%d] halo=%d n_rows=%d weight_count=%ld max_threads=%d\n",
            n, n_local, sigma, support, disp_min, disp_max, halo,
            kernel->n_rows, weight_count, max_threads());
    }
}

/*
 * Charge assignment leaves most grid nodes exactly zero for typical P3M
 * systems.  Scattering the compact kernel from those non-zero sources avoids
 * evaluating the stencil at every output node.  The operation is algebraically
 * identical to the dense gather below.  With OpenMP, contiguous ranges of
 * sources accumulate into private x-slab buffers, including only the output
 * planes reachable by each range.  A final deterministic reduction avoids
 * atomics and random concurrent writes without replicating the full grid for
 * every thread.
 */
static int smooth_charges_wendland_sparse(grid *grid, wendland_nofft_kernel *kernel) {
    const int n = kernel->n;
    const int n_local = kernel->n_local;
    const int halo = kernel->halo;
    const long n2 = (long)n * n;
    const long ext_size = (long)(n_local + 2 * halo) * n2;
    const long output_size = (long)n_local * n2;
    long nonzero_count = 0;

    /*
     * Scanning ext_size once, serially, would dwarf the sparse convolution
     * below for typical P3M charge assignments (few nonzeros over a full
     * grid). Split it into contiguous per-thread chunks, count locally, then
     * write with a prefix-sum offset so the result stays ordered by p, which
     * the work-balancing below relies on.
     */
    int scan_threads = max_threads();
    if (scan_threads > kernel->thread_capacity) scan_threads = kernel->thread_capacity;
    if (scan_threads < 1) scan_threads = 1;
    if ((long)scan_threads > ext_size) scan_threads = ext_size > 0 ? (int)ext_size : 1;
    if (scan_threads > 1) {
        long scan_counts[scan_threads];
        const long chunk = (ext_size + scan_threads - 1) / scan_threads;
        #pragma omp parallel num_threads(scan_threads)
        {
            const int tid = thread_number();
            long begin = (long)tid * chunk;
            long end = begin + chunk;
            if (begin > ext_size) begin = ext_size;
            if (end > ext_size) end = ext_size;
            long local = 0;
            for (long p = begin; p < end; ++p) {
                const int source_i = p / n2;
                if (kernel->extended[p] != 0.0 && kernel->source_work[source_i] > 0) ++local;
            }
            scan_counts[tid] = local;
            #pragma omp barrier
            #pragma omp single
            {
                long offset = 0;
                for (int t = 0; t < scan_threads; ++t) {
                    long c = scan_counts[t];
                    scan_counts[t] = offset;
                    offset += c;
                }
                nonzero_count = offset;
            }
            long write = scan_counts[tid];
            for (long p = begin; p < end; ++p) {
                const int source_i = p / n2;
                if (kernel->extended[p] != 0.0 && kernel->source_work[source_i] > 0) {
                    kernel->nonzero_indices[write++] = p;
                }
            }
        }
    } else {
        for (long p = 0; p < ext_size; ++p) {
            const int source_i = p / n2;
            if (kernel->extended[p] != 0.0 && kernel->source_work[source_i] > 0) {
                kernel->nonzero_indices[nonzero_count++] = p;
            }
        }
    }
    /* Random scatter writes stop winning well before a 50% fill fraction. */
    int sparse_ok = nonzero_count * 5 < output_size;
    /* Benchmark-only override to measure both paths at the same (N, fill
     * fraction) instead of only whichever the fixed threshold happens to
     * pick; unset in normal use, so default behaviour is untouched. */
    const char *force_path = getenv("MAZE_WENDLAND_FORCE_PATH");
    if (force_path != NULL) {
        if (strcmp(force_path, "dense") == 0) sparse_ok = 0;
        else if (strcmp(force_path, "sparse") == 0) sparse_ok = 1;
    }
    if (getenv("MAZE_WENDLAND_DEBUG") != NULL) {
        mpi_fprintf(stderr,
            "[wendland-debug] call: nonzero_count=%ld output_size=%ld fill_frac=%.6f path=%s%s\n",
            nonzero_count, output_size, (double)nonzero_count / (double)output_size,
            sparse_ok ? "sparse" : "dense-fallback", force_path ? " (forced)" : "");
    }
    if (!sparse_ok) return 0;

    int n_threads = max_threads();
    if (n_threads > kernel->thread_capacity) n_threads = kernel->thread_capacity;
    if (n_threads > nonzero_count) n_threads = (int)nonzero_count;
    if (n_threads < 2) {
        memset(kernel->output, 0, output_size * sizeof(double));
        for (long nz = 0; nz < nonzero_count; ++nz) {
            const long p = kernel->nonzero_indices[nz];
            const int source_i = p / n2;
            const long rem = p - (long)source_i * n2;
            const int source_j = rem / n;
            const int source_k = rem - (long)source_j * n;
            const double charge = kernel->extended[p];

            int first_dx = halo - source_i;
            int last_dx = halo - source_i + n_local - 1;
            if (first_dx < kernel->disp_min) first_dx = kernel->disp_min;
            if (last_dx > kernel->disp_max) last_dx = kernel->disp_max;
            for (int dx = first_dx; dx <= last_dx; ++dx) {
                const int output_i = source_i - halo + dx;
                const int r_begin = kernel->dx_row_first[dx - kernel->disp_min];
                const int r_end = kernel->dx_row_first[dx - kernel->disp_min + 1];
                for (int r = r_begin; r < r_end; ++r) {
                    const wendland_row *row = kernel->rows + r;
                    int output_j = source_j + row->dy;
                    if (output_j < 0) output_j += n;
                    if (output_j >= n) output_j -= n;
                    const long line = (long)output_i * n2 + (long)output_j * n;
                    int output_k = source_k + row->dz_first - row->dz_count + 1;
                    if (output_k < 0) output_k += n;
                    if (output_k >= n) output_k -= n;
                    int first_count = row->dz_count;
                    if (first_count > n - output_k) first_count = n - output_k;
                    const double *weights = kernel->scatter_weights + row->weight_first;
                    #pragma omp simd
                    for (int z = 0; z < first_count; ++z) {
                        kernel->output[line + output_k + z] += charge * weights[z];
                    }
                    #pragma omp simd
                    for (int z = first_count; z < row->dz_count; ++z) {
                        kernel->output[line + z - first_count] += charge * weights[z];
                    }
                }
            }
        }
        memcpy(grid->q, kernel->output, output_size * sizeof(double));
        return 1;
    }

    /* Balance by useful coefficients: halo sources touch fewer local planes. */
    long total_work = 0;
    for (long nz = 0; nz < nonzero_count; ++nz) {
        total_work += kernel->source_work[kernel->nonzero_indices[nz] / n2];
    }
    long nz_cursor = 0;
    long cumulative_work = 0;
    for (int t = 0; t < n_threads; ++t) {
        kernel->thread_nz_begin[t] = nz_cursor;
        if (t == n_threads - 1) {
            nz_cursor = nonzero_count;
        } else {
            const long target = total_work * (t + 1) / n_threads;
            const long latest_end = nonzero_count - (n_threads - t - 1);
            do {
                cumulative_work += kernel->source_work[
                    kernel->nonzero_indices[nz_cursor] / n2];
                ++nz_cursor;
            } while (nz_cursor < latest_end && cumulative_work < target);
        }
        kernel->thread_nz_end[t] = nz_cursor;
    }

    /* Prepare the smallest x-slab buffer that can receive each source range. */
    for (int t = 0; t < n_threads; ++t) {
        const long nz_begin = kernel->thread_nz_begin[t];
        const long nz_end = kernel->thread_nz_end[t];
        const int first_source_i = kernel->nonzero_indices[nz_begin] / n2;
        const int last_source_i = kernel->nonzero_indices[nz_end - 1] / n2;
        int output_begin = first_source_i - halo + kernel->disp_min;
        int output_end = last_source_i - halo + kernel->disp_max + 1;
        if (output_begin < 0) output_begin = 0;
        if (output_end > n_local) output_end = n_local;
        kernel->thread_output_begin[t] = output_begin;
        kernel->thread_output_end[t] = output_end;
        const long required = (long)(output_end - output_begin) * n2;
        if (required > kernel->thread_buffer_capacity[t]) {
            double *resized = realloc(kernel->thread_buffers[t], required * sizeof(double));
            if (resized == NULL) {
                mpi_fprintf(stderr, "Unable to allocate no-FFT Wendland thread buffer\n");
                exit(EXIT_FAILURE);
            }
            kernel->thread_buffers[t] = resized;
            kernel->thread_buffer_capacity[t] = required;
        }
    }

#ifdef _OPENMP
    const int dynamic_threads = omp_get_dynamic();
    if (dynamic_threads) omp_set_dynamic(0);
#endif
    #pragma omp parallel num_threads(n_threads)
    {
        const int tid = thread_number();
        const long nz_begin = kernel->thread_nz_begin[tid];
        const long nz_end = kernel->thread_nz_end[tid];
        const int output_begin = kernel->thread_output_begin[tid];
        const int output_end = kernel->thread_output_end[tid];
        double *private_output = kernel->thread_buffers[tid];
        memset(private_output, 0, (long)(output_end - output_begin) * n2 * sizeof(double));

        for (long nz = nz_begin; nz < nz_end; ++nz) {
            const long p = kernel->nonzero_indices[nz];
            const int source_i = p / n2;
            const long rem = p - (long)source_i * n2;
            const int source_j = rem / n;
            const int source_k = rem - (long)source_j * n;
            const double charge = kernel->extended[p];

            int first_dx = halo - source_i;
            int last_dx = halo - source_i + n_local - 1;
            if (first_dx < kernel->disp_min) first_dx = kernel->disp_min;
            if (last_dx > kernel->disp_max) last_dx = kernel->disp_max;
            for (int dx = first_dx; dx <= last_dx; ++dx) {
                const int output_i = source_i - halo + dx;
                const int r_begin = kernel->dx_row_first[dx - kernel->disp_min];
                const int r_end = kernel->dx_row_first[dx - kernel->disp_min + 1];
                for (int r = r_begin; r < r_end; ++r) {
                    const wendland_row *row = kernel->rows + r;
                    int output_j = source_j + row->dy;
                    if (output_j < 0) output_j += n;
                    if (output_j >= n) output_j -= n;
                    const long line = (long)(output_i - output_begin) * n2
                                    + (long)output_j * n;
                    int output_k = source_k + row->dz_first - row->dz_count + 1;
                    if (output_k < 0) output_k += n;
                    if (output_k >= n) output_k -= n;
                    int first_count = row->dz_count;
                    if (first_count > n - output_k) first_count = n - output_k;
                    const double *weights = kernel->scatter_weights + row->weight_first;
                    #pragma omp simd
                    for (int z = 0; z < first_count; ++z) {
                        private_output[line + output_k + z] += charge * weights[z];
                    }
                    #pragma omp simd
                    for (int z = first_count; z < row->dz_count; ++z) {
                        private_output[line + z - first_count] += charge * weights[z];
                    }
                }
            }
        }

        /*
         * thread_output_begin/end are non-decreasing across t (threads
         * consume nonzero indices in increasing source_i order), so the
         * threads overlapping a given output plane form a contiguous range
         * found by binary search, instead of scanning all n_threads per
         * output element below.
         */
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (int output_i = 0; output_i < n_local; ++output_i) {
            int lo = 0, hi = n_threads;
            {
                int l = 0, r = n_threads;
                while (l < r) {
                    int m = (l + r) / 2;
                    if (kernel->thread_output_end[m] > output_i) r = m; else l = m + 1;
                }
                lo = l;
            }
            {
                int l = 0, r = n_threads;
                while (l < r) {
                    int m = (l + r) / 2;
                    if (kernel->thread_output_begin[m] <= output_i) l = m + 1; else r = m;
                }
                hi = l;
            }
            double *dst = kernel->output + (long)output_i * n2;
            if (lo >= hi) {
                memset(dst, 0, n2 * sizeof(double));
            } else if (hi - lo == 1) {
                const double *src = kernel->thread_buffers[lo] +
                    (long)(output_i - kernel->thread_output_begin[lo]) * n2;
                memcpy(dst, src, n2 * sizeof(double));
            } else {
                memset(dst, 0, n2 * sizeof(double));
                for (int t = lo; t < hi; ++t) {
                    const double *src = kernel->thread_buffers[t] +
                        (long)(output_i - kernel->thread_output_begin[t]) * n2;
                    #pragma omp simd
                    for (long yz = 0; yz < n2; ++yz) dst[yz] += src[yz];
                }
            }
        }
    }
#ifdef _OPENMP
    if (dynamic_threads) omp_set_dynamic(1);
#endif

    memcpy(grid->q, kernel->output, output_size * sizeof(double));
    return 1;
}

void smooth_charges_wendland_nofft(grid *grid) {
    wendland_nofft_kernel *kernel = grid->smoothing_kernel;
    const int n = kernel->n;
    const int n_local = kernel->n_local;
    const int halo = kernel->halo;
    const long n2 = (long)n * n;

    if (n_local == 0) return;

    fill_extended_grid(grid, kernel);
    if (smooth_charges_wendland_sparse(grid, kernel)) return;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n_local; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                double value = 0.0;
                for (int r = 0; r < kernel->n_rows; ++r) {
                    const wendland_row *row = kernel->rows + r;
                    const long plane = (long)(halo + i - row->dx) * n2;
                    int source_y = j - row->dy;
                    if (source_y < 0) source_y += n;
                    if (source_y >= n) source_y -= n;
                    const long line = plane + (long)source_y * n;
                    const double *weights = kernel->weights + row->weight_first;
                    int source_z = k - row->dz_first;
                    if (source_z < 0) source_z += n;
                    if (source_z >= n) source_z -= n;
                    int first_count = row->dz_count;
                    if (first_count > n - source_z) first_count = n - source_z;
                    const double *source = kernel->extended + line + source_z;
                    #pragma omp simd reduction(+:value)
                    for (int z = 0; z < first_count; ++z) {
                        value += weights[z] * source[z];
                    }
                    source = kernel->extended + line;
                    #pragma omp simd reduction(+:value)
                    for (int z = first_count; z < row->dz_count; ++z) {
                        value += weights[z] * source[z - first_count];
                    }
                }
                kernel->output[(long)i * n2 + (long)j * n + k] = value;
            }
        }
    }
    memcpy(grid->q, kernel->output, (long)n_local * n2 * sizeof(double));
}

void smooth_charges_wendland_nofft_free(grid *grid) {
    wendland_nofft_kernel *kernel = grid->smoothing_kernel;
    if (kernel == NULL) return;
    free(kernel->rows);
    free(kernel->dx_row_first);
    free(kernel->weights);
    free(kernel->scatter_weights);
    free(kernel->extended);
    free(kernel->output);
    free(kernel->nonzero_indices);
    for (int t = 0; t < kernel->thread_capacity; ++t) free(kernel->thread_buffers[t]);
    free(kernel->thread_output_begin);
    free(kernel->thread_output_end);
    free(kernel->thread_nz_begin);
    free(kernel->thread_nz_end);
    free(kernel->thread_buffer_capacity);
    free(kernel->thread_buffers);
    free(kernel->source_work);
    free(kernel);
    grid->smoothing_kernel = NULL;
}
