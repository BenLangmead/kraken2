/*
 * Phase 0 instrumentation for the GPU acceleration effort.  See k2_profile.h.
 */

#include "k2_profile.h"

namespace kraken2 {

thread_local volatile uint64_t k2_ablation_sink = 0;

const char *K2AblationLevelName() {
  switch (K2_ABLATE) {
    case 0:  return "L0 full";
    case 1:  return "L1 -format";
    case 2:  return "L2 -format -hll";
    case 3:  return "L3 -format -hll -resolve";
    case 4:  return "L4 -format -hll -resolve -counters";
    case 5:  return "L5 -format -hll -resolve -counters -probe";
    case 6:  return "L6 -format -hll -resolve -counters -probe -scan (parse floor)";
    default: return "L? unknown ablation level";
  }
}

#ifdef K2_PROFILE

double k2_region_seconds[K2_MAX_THREADS][K2_REGION_COUNT] = {{0.0}};

void K2ReportRegions(double wall_seconds) {
  static const char *names[K2_REGION_COUNT] = { "parse", "classify", "output" };
  double totals[K2_REGION_COUNT] = {0.0};
  for (int t = 0; t < K2_MAX_THREADS; t++)
    for (int r = 0; r < K2_REGION_COUNT; r++)
      totals[r] += k2_region_seconds[t][r];

  double cpu_total = 0.0;
  for (int r = 0; r < K2_REGION_COUNT; r++)
    cpu_total += totals[r];

  fprintf(stderr, "\n  region breakdown (summed over threads)\n");
  for (int r = 0; r < K2_REGION_COUNT; r++) {
    fprintf(stderr, "    %-10s %10.3f s  %6.2f%% of accounted thread time\n",
            names[r], totals[r],
            cpu_total > 0.0 ? 100.0 * totals[r] / cpu_total : 0.0);
  }
  fprintf(stderr, "    %-10s %10.3f s\n", "accounted", cpu_total);
  fprintf(stderr, "    %-10s %10.3f s\n", "wall", wall_seconds);
}

#endif  // K2_PROFILE

}  // end namespace
