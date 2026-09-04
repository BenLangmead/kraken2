/*
 * Phase 0 instrumentation for the GPU acceleration effort.
 *
 * Two independent instruments live here.
 *
 * 1. A cumulative ablation ladder selected by K2_ABLATE.  Each level compiles
 *    out one more region of the per-read pipeline, so the difference in run
 *    time between two adjacent levels is that region's cost.  Nothing is timed
 *    from inside the hot loop, so the measurement does not perturb what it
 *    measures.
 *
 *      0  everything (stock behavior)
 *      1  no per-read output construction (hitlist string and FASTQ records)
 *      2  ... and no HyperLogLog k-mer registration
 *      3  ... and no ResolveTree call
 *      4  ... and no hit_counts / taxa accumulation
 *      5  ... and no hash table lookup
 *      6  ... and no minimizer scan
 *
 *    Level 6 is the parse floor.  Writing L_n for the run time at level n:
 *
 *      offloadable regions (scan + probe + per-read hit accumulation) = L3 - L6
 *      tuned CPU baseline, conservatively                             = L2
 *      offloadable fraction  f = (L3 - L6) / L2
 *      speedup ceiling         = L2 / (L2 - L3 + L6)
 *
 *    L2 stands in for the tuned baseline because gating HyperLogLog on -K and
 *    dropping output construction under -O - are semantics-preserving wins that
 *    the tuned CPU arm gets for free.  It ignores any parse improvements, which
 *    makes the resulting ceiling an underestimate.
 *
 * 2. Coarse per-region wall timers enabled by K2_PROFILE, accumulated per
 *    thread and reported at exit.  These are a cross-check on the ladder, not
 *    the primary instrument, and are only taken once per batch or once per
 *    read so their overhead stays negligible.
 */

#ifndef KRAKEN2_K2_PROFILE_H_
#define KRAKEN2_K2_PROFILE_H_

#include <cstdint>

#ifndef K2_ABLATE
#define K2_ABLATE 0
#endif

#define K2_HAVE_FORMAT   (K2_ABLATE < 1)
#define K2_HAVE_HLL      (K2_ABLATE < 2)
#define K2_HAVE_RESOLVE  (K2_ABLATE < 3)
#define K2_HAVE_COUNTERS (K2_ABLATE < 4)
#define K2_HAVE_PROBE    (K2_ABLATE < 5)
#define K2_HAVE_SCAN     (K2_ABLATE < 6)

namespace kraken2 {

// Keeps ablated-away reads from being optimized out entirely: at level 6 the
// scanner is replaced by a traversal that still touches every sequence byte,
// so the parse floor includes the cost of streaming the read data.
//
// Thread-local, because a shared volatile counter would put every thread on one
// cache line and turn the parse floor into a contention benchmark.
extern thread_local volatile uint64_t k2_ablation_sink;

const char *K2AblationLevelName();

}  // end namespace

#ifdef K2_PROFILE

#include <chrono>
#include <cstdio>
#include "omp_hack.h"

namespace kraken2 {

enum K2Region {
  K2_REGION_PARSE = 0,
  K2_REGION_CLASSIFY,
  K2_REGION_OUTPUT,
  K2_REGION_COUNT
};

const int K2_MAX_THREADS = 512;
extern double k2_region_seconds[K2_MAX_THREADS][K2_REGION_COUNT];

class K2Timer {
  public:
  explicit K2Timer(K2Region region)
      : region_(region), start_(std::chrono::steady_clock::now()) { }
  ~K2Timer() {
    std::chrono::duration<double> elapsed =
        std::chrono::steady_clock::now() - start_;
    int tid = omp_get_thread_num();
    if (tid >= 0 && tid < K2_MAX_THREADS)
      k2_region_seconds[tid][region_] += elapsed.count();
  }
  private:
  K2Region region_;
  std::chrono::steady_clock::time_point start_;
};

void K2ReportRegions(double wall_seconds);

}  // end namespace

#define K2_TIME_REGION(r) kraken2::K2Timer k2_timer_##r(kraken2::K2_REGION_##r)

#else  // ! K2_PROFILE

#define K2_TIME_REGION(r) do { } while (0)

namespace kraken2 {
inline void K2ReportRegions(double) { }
}

#endif  // K2_PROFILE

#endif  // KRAKEN2_K2_PROFILE_H_
