#include "omp_base.h"

#ifdef _OPENMP
#include <stdlib.h>

#ifdef __clang__
/* LLVM libomp extension; weak so that other OpenMP runtimes still link */
extern void kmp_set_blocktime(int) __attribute__((weak));
#endif

/*
 * With libomp the idle threads go to sleep right after a parallel region, and waking them
 * up dominates on the small grids of the multigrid hierarchy (measured ~1.7x slower).
 * Keep them spinning for 1 s unless the user chose a wait policy explicitly.
 */
static void omp_tune_wait_policy(void) {
    static int done = 0;
    if (done) return;
    done = 1;
#ifdef __clang__
    if (getenv("KMP_BLOCKTIME") == NULL && getenv("OMP_WAIT_POLICY") == NULL && kmp_set_blocktime != NULL) {
        kmp_set_blocktime(1000);
    }
#endif
}

int get_omp_thread_num() {
    return omp_get_thread_num();
}

int get_omp_max_threads() {
    omp_tune_wait_policy();
    return omp_get_max_threads();
}

#else // _OPENMP

int get_omp_thread_num() {
    return 0;
}

int get_omp_max_threads() {
    return 0;
}

#endif // _OPENMP