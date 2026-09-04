#!/usr/bin/env python3
"""Run the Phase 0 ablation ladder and report the speedup ceiling.

Times the classification region only, by parsing the "processed in N.NNNs" line
that classify already prints -- its timer starts after the database is loaded,
so database load time is excluded and does not dilute the measurement.

Reported quantities:

    region costs   L_{n-1} - L_n for each adjacent pair
    offloadable    L3 - L6   (minimizer scan + table probe + hit accumulation)
    tuned baseline L2        (stock minus output construction and HyperLogLog,
                              both semantics-preserving wins the tuned CPU arm
                              gets for free; ignores parse improvements, so the
                              resulting ceiling is an underestimate)
    f              (L3 - L6) / L2
    ceiling        1 / (1 - f) = L2 / (L2 - L3 + L6)
"""

import argparse
import os
import random
import re
import statistics
import subprocess
import sys
import time

LEVELS = [
    (0, "full"),
    (1, "-format"),
    (2, "-hll"),
    (3, "-resolve"),
    (4, "-counters"),
    (5, "-probe"),
    (6, "-scan"),
]

# What the delta between level n-1 and n isolates.
REGION_OF = {
    1: "output construction",
    2: "HyperLogLog registration",
    3: "ResolveTree",
    4: "hit_counts + taxa",
    5: "hash table probe",
    6: "minimizer scan",
}

TIME_RE = re.compile(r"processed in ([0-9.]+)s")
CLASS_RE = re.compile(r"([0-9]+) sequences classified \(([0-9.]+)%\)")


def run_one(binary, db, r1, r2, threads, report, extra, mmap_db=True):
    cmd = [binary,
           "-H", os.path.join(db, "hash.k2d"),
           "-t", os.path.join(db, "taxo.k2d"),
           "-o", os.path.join(db, "opts.k2d"),
           "-R", report, "-O", "-", "-p", str(threads)]
    if mmap_db:
        # Map the table instead of reading 16 GB into heap on every run; with
        # 22 runs on a 48 GB machine the repeated allocation evicts the page
        # cache holding the reads and swamps the signal.
        cmd += ["-M"]
    if r2:
        cmd += ["-P"]
    cmd += extra + [r1] + ([r2] if r2 else [])
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, check=True)
    wall = time.time() - t0
    err = p.stderr.decode("utf-8", "replace")
    m = TIME_RE.search(err)
    if not m:
        sys.exit("could not parse timing from:\n" + err[-2000:])
    c = CLASS_RE.search(err)
    return float(m.group(1)), wall, (float(c.group(2)) if c else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default=os.path.expanduser("~/kraken2-gpu-work/bin"))
    ap.add_argument("--db", required=True)
    ap.add_argument("--r1", required=True)
    ap.add_argument("--r2", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=os.path.expanduser("~/kraken2-gpu-work/results"))
    ap.add_argument("--label", default="phase0")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--cooldown", type=float, default=3.0,
                    help="seconds to idle between runs, to limit thermal drift")
    ap.add_argument("--no-mmap", action="store_true",
                    help="read the table into heap instead of mapping it")
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    for lvl, _ in LEVELS:
        b = os.path.join(args.bin_dir, "classify_L%d" % lvl)
        if not os.path.exists(b):
            sys.exit("missing %s -- run scripts/phase0_build.sh" % b)

    def one(lvl):
        return run_one(os.path.join(args.bin_dir, "classify_L%d" % lvl),
                       args.db, args.r1, args.r2, args.threads,
                       os.path.join(args.out, "%s_L%d.report" % (args.label, lvl)),
                       args.extra, mmap_db=not args.no_mmap)

    # Warm the page cache and fault in the mapped table once, so no single
    # level pays for cold reads.
    print("warming page cache...", file=sys.stderr)
    one(0)

    # Interleave levels within each repetition, in a fresh random order each
    # time.  Running all reps of one level before moving on lets thermal drift
    # over a long session masquerade as a real cost difference between levels,
    # and on a laptop under sustained load it certainly will.
    runs = {lvl: [] for lvl, _ in LEVELS}
    pct = {}
    raw_path = os.path.join(args.out, "%s_raw.csv" % args.label)
    with open(raw_path, "w") as rf:
        rf.write("rep,level,seconds,classified_pct\n")
    order = [lvl for lvl, _ in LEVELS]
    for rep in range(args.reps):
        rng.shuffle(order)
        for lvl in order:
            secs, _wall, cpct = one(lvl)
            runs[lvl].append(secs)
            pct[lvl] = cpct
            # Emit every measurement as it lands so a killed run still leaves
            # usable data behind.
            print("  rep %d  L%d  %8.3fs  %6.2f%% classified"
                  % (rep + 1, lvl, secs, cpct), file=sys.stderr)
            with open(raw_path, "a") as rf:
                rf.write("%d,%d,%.4f,%.3f\n" % (rep + 1, lvl, secs, cpct))
            if args.cooldown:
                time.sleep(args.cooldown)
        print("  -- rep %d/%d done --" % (rep + 1, args.reps), file=sys.stderr)

    # Minimum is the estimator: every source of interference here (throttling,
    # page cache eviction, other processes) adds time, never removes it.
    t = {lvl: min(runs[lvl]) for lvl, _ in LEVELS}

    print("\n%-4s %-12s %10s %10s %10s   %s"
          % ("lvl", "removes", "min s", "median s", "spread", "classified"),
          file=sys.stderr)
    worst_spread = 0.0
    for lvl, name in LEVELS:
        med = statistics.median(runs[lvl])
        spread = (max(runs[lvl]) - t[lvl]) / max(t[lvl], 1e-9)
        worst_spread = max(worst_spread, spread)
        print("%-4s %-12s %10.3f %10.3f %9.1f%%   %8.2f%%"
              % ("L%d" % lvl, name, t[lvl], med, 100 * spread, pct[lvl]),
              file=sys.stderr)

    # A cumulative ablation ladder must be monotonically non-increasing: each
    # level only removes work.  Any inversion means the measurement is
    # contaminated, and no ceiling computed from it is trustworthy.
    def spread_of(lvl):
        return (max(runs[lvl]) - min(runs[lvl])) / max(min(runs[lvl]), 1e-9)

    # An inversion smaller than the measured noise is a tie, not a violation.
    # A level whose region is already skipped at run time (gated HyperLogLog or
    # hitlist construction) legitimately ties with the level above it.
    inversions = []
    for lvl in range(1, 7):
        tol = max(0.01, spread_of(lvl), spread_of(lvl - 1))
        if t[lvl] > t[lvl - 1] * (1.0 + tol):
            inversions.append((lvl, t[lvl - 1], t[lvl]))
    ties = [lvl for lvl in range(1, 7)
            if abs(t[lvl] - t[lvl - 1]) <= t[lvl - 1] * max(0.01, spread_of(lvl))]
    if inversions:
        print("\n  INVALID: ladder is not monotonic -- each level only removes work,",
              file=sys.stderr)
        print("  so times must be non-increasing.  Inversions:", file=sys.stderr)
        for lvl, prev, cur in inversions:
            print("    L%d (%.3fs) > L%d (%.3fs)  [%s]"
                  % (lvl, cur, lvl - 1, prev, REGION_OF[lvl]), file=sys.stderr)
        print("  Re-run on a quiet machine, or raise --reps.  Not reporting a ceiling.",
              file=sys.stderr)
        sys.exit(2)
    if worst_spread > 0.15:
        print("\n  WARNING: worst run-to-run spread %.0f%% exceeds 15%%; treat the"
              % (100 * worst_spread), file=sys.stderr)
        print("  ceiling below as indicative only.", file=sys.stderr)

    if ties:
        print("\n  note: %s tie with the level above, i.e. that region is already"
              % ", ".join("L%d" % l for l in ties), file=sys.stderr)
        print("  skipped at run time in this configuration.", file=sys.stderr)

    print("\nregion costs (min seconds, and share of L0)", file=sys.stderr)
    for lvl in range(1, 7):
        cost = t[lvl - 1] - t[lvl]
        print("  %-26s %8.3f s   %6.2f%%"
              % (REGION_OF[lvl], cost, 100.0 * cost / t[0]), file=sys.stderr)
    print("  %-26s %8.3f s   %6.2f%%"
          % ("parse floor (L6)", t[6], 100.0 * t[6] / t[0]), file=sys.stderr)

    offloadable = t[3] - t[6]
    # L0 is the tuned baseline: in this configuration format and HyperLogLog
    # are already skipped at run time, so L1 and L2 have nothing to remove.
    tuned = t[0]
    f = offloadable / tuned
    ceiling = (1.0 / (1.0 - f)) if f < 1.0 else float("inf")

    print("\n%s" % ("=" * 62), file=sys.stderr)
    print("  offloadable (scan + probe + hits) = L3 - L6 = %8.3f s" % offloadable,
          file=sys.stderr)
    print("  tuned CPU baseline               = L0      = %8.3f s" % tuned,
          file=sys.stderr)
    print("  offloadable fraction f                     = %8.4f" % f, file=sys.stderr)
    print("  speedup ceiling  1/(1-f)                   = %8.2fx" % ceiling,
          file=sys.stderr)
    print("%s" % ("=" * 62), file=sys.stderr)

    verdict = ("PASS -- comfortable margin" if ceiling >= 2.5 else
               "MARGINAL -- 2x reachable but with no headroom" if ceiling >= 2.0 else
               "FAIL -- 2x is unreachable on this database; redesign")
    print("  gate (>= 2.5x): %s\n" % verdict, file=sys.stderr)

    csv = os.path.join(args.out, "%s_ladder.csv" % args.label)
    with open(csv, "w") as fh:
        fh.write("level,removes,median_s,classified_pct\n")
        for lvl, name in LEVELS:
            fh.write("L%d,%s,%.4f,%.3f\n" % (lvl, name, t[lvl], pct[lvl]))
        fh.write("#f,%.6f\n#ceiling,%.4f\n" % (f, ceiling))
    print("  wrote %s" % csv, file=sys.stderr)


if __name__ == "__main__":
    main()
