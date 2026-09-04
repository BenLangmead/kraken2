#!/usr/bin/env python3
"""Simulate a metagenomic PE 2x150 read set for Kraken 2 benchmarking.

Draws a log-normal abundance profile over the in-db genome pool, assigns the
held-out pool a fixed share of the reads so the classified fraction lands near
a chosen target, runs art_illumina once per genome in parallel, then interleaves the per-genome
FASTQ files at record granularity.

The interleave matters as much as the simulation.  Concatenating per-genome
blocks would put tens of thousands of consecutive reads on the same genome,
which clusters hash table accesses and makes the CPU arm look better than it is
on real data.  The draw order is materialized as a shuffled array holding one
entry per record, so each genome contributes exactly its share and the merge
costs one indexed lookup per record rather than a scan over every source.

Everything is seeded, so a given (seed, target, pair count) reproduces exactly.
"""

import argparse
import array
import gzip
import os
import random
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def genome_length(path):
    """Total residue count of a (possibly gzipped) FASTA."""
    opener = gzip.open if path.endswith(".gz") else open
    total = 0
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.startswith(">"):
                total += len(line.rstrip())
    return total


def decompress(src, dst):
    with gzip.open(src, "rb") as i, open(dst, "wb") as o:
        shutil.copyfileobj(i, o, 1 << 20)
    return dst


def run_art(art, fasta, out_prefix, pairs, glen, readlen, seed, log):
    """Fold coverage that yields approximately `pairs` read pairs."""
    fold = pairs * 2.0 * readlen / max(glen, 1)
    if pairs < 10 or fold <= 0:
        return None
    cmd = [art, "-ss", "HS25", "-i", fasta, "-p", "-l", str(readlen),
           "-f", "%.6f" % fold, "-m", "350", "-s", "50",
           "-o", out_prefix, "-na", "-rs", str(seed)]
    with open(log, "wb") as lg:
        subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, check=True)
    r1, r2 = out_prefix + "1.fq", out_prefix + "2.fq"
    return (r1, r2) if os.path.exists(r1) and os.path.exists(r2) else None


class FastqPairSource:
    """Streams 4-line records from a mate pair, tracking how many remain."""

    __slots__ = ("f1", "f2", "remaining", "tag")

    def __init__(self, r1, r2, count, tag):
        self.f1 = open(r1, "r", buffering=1 << 20)
        self.f2 = open(r2, "r", buffering=1 << 20)
        self.remaining = count
        self.tag = tag

    def next_record(self):
        a = [self.f1.readline() for _ in range(4)]
        b = [self.f2.readline() for _ in range(4)]
        if not a[0] or not b[0]:
            self.remaining = 0
            return None
        self.remaining -= 1
        return "".join(a), "".join(b)

    def close(self):
        self.f1.close()
        self.f2.close()


def count_records(path):
    """Record count via wc -l; orders of magnitude faster than a Python loop."""
    out = subprocess.run(["wc", "-l", path], capture_output=True, check=True)
    return int(out.stdout.split()[0]) // 4


def weighted_merge(sources, out1, out2, rng, truth_path):
    """Interleave records at record granularity in one streaming pass.

    The draw order is materialized up front as a shuffled array of source
    indices -- one entry per record each source holds -- so the inner loop is a
    single indexed lookup rather than a scan over every source.  That keeps the
    merge linear in the number of records instead of records times sources,
    which for a few hundred genomes is the difference between minutes and hours.
    """
    order = array.array("i")
    for i, src in enumerate(sources):
        order.extend(array.array("i", [i]) * src.remaining)
    total = len(order)
    print("  shuffling draw order for %d pairs..." % total, file=sys.stderr)
    rng.shuffle(order)

    written = 0
    with open(out1, "w", buffering=1 << 22) as o1, \
         open(out2, "w", buffering=1 << 22) as o2, \
         open(truth_path, "w", buffering=1 << 20) as tr:
        w1, w2, wt = o1.write, o2.write, tr.write
        for idx in order:
            rec = sources[idx].next_record()
            if rec is None:
                continue
            w1(rec[0])
            w2(rec[1])
            wt(sources[idx].tag)
            written += 1
            if written % 2000000 == 0:
                print("  merged %d / %d pairs" % (written, total), file=sys.stderr)
    for s in sources:
        s.close()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.expanduser("~/kraken2-gpu-work"))
    ap.add_argument("--pairs", type=int, default=10_000_000,
                    help="total read pairs to emit")
    ap.add_argument("--held-out-frac", type=float, default=0.25,
                    help="share of reads drawn from genomes absent from the database")
    ap.add_argument("--readlen", type=int, default=150)
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="log-normal sigma for the abundance profile")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--name", default="sim")
    args = ap.parse_args()

    work = args.work
    art = os.path.expanduser("~/miniforge3/envs/k2sim/bin/art_illumina")
    if not os.path.exists(art):
        sys.exit("art_illumina not found at %s" % art)

    gen = os.path.join(work, "genomes")
    tmp = os.path.join(work, "simtmp")
    out = os.path.join(work, "reads")
    for d in (tmp, out):
        os.makedirs(d, exist_ok=True)

    pools = {}
    for pool in ("in-db", "held-out"):
        d = os.path.join(gen, pool)
        pools[pool] = sorted(
            os.path.join(d, f) for f in os.listdir(d) if f.endswith(".fna.gz"))
        if not pools[pool]:
            sys.exit("no genomes in %s" % d)
    print("in-db genomes: %d   held-out genomes: %d"
          % (len(pools["in-db"]), len(pools["held-out"])), file=sys.stderr)

    rng = random.Random(args.seed)

    # ---- abundance profile ------------------------------------------------
    plan = []  # (fasta_gz, pairs, pool)
    for pool, share in (("in-db", 1.0 - args.held_out_frac),
                        ("held-out", args.held_out_frac)):
        files = pools[pool]
        w = [rng.lognormvariate(0.0, args.sigma) for _ in files]
        tot = sum(w)
        budget = args.pairs * share
        for f, wi in zip(files, w):
            plan.append((f, int(budget * wi / tot), pool))

    print("measuring genome lengths...", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        lengths = list(ex.map(lambda p: genome_length(p[0]), plan))

    # ---- simulate ---------------------------------------------------------
    def job(i):
        fasta_gz, pairs, pool = plan[i]
        if pairs < 10:
            return None
        stem = os.path.basename(fasta_gz).replace(".fna.gz", "")
        flat = os.path.join(tmp, stem + ".fna")
        if not os.path.exists(flat):
            decompress(fasta_gz, flat)
        res = run_art(art, flat, os.path.join(tmp, stem + "_"), pairs,
                      lengths[i], args.readlen, args.seed + i,
                      os.path.join(tmp, stem + ".log"))
        os.remove(flat)
        return (res, pool, stem) if res else None

    print("simulating with art_illumina (%d jobs)..." % args.jobs, file=sys.stderr)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = [r for r in ex.map(job, range(len(plan))) if r]
    print("simulated %d genomes" % len(results), file=sys.stderr)

    # ---- merge ------------------------------------------------------------
    print("counting and merging...", file=sys.stderr)
    sources = []
    for (r1, r2), pool, stem in results:
        n = count_records(r1)
        if n > 0:
            sources.append(FastqPairSource(r1, r2, n, "%s\t%s\n" % (pool, stem)))

    o1 = os.path.join(out, "%s_1.fq" % args.name)
    o2 = os.path.join(out, "%s_2.fq" % args.name)
    truth = os.path.join(out, "%s_truth.tsv" % args.name)
    written = weighted_merge(sources, o1, o2, rng, truth)

    for (r1, r2), _, _ in results:
        for f in (r1, r2):
            if os.path.exists(f):
                os.remove(f)

    held = sum(1 for (_, pool, _) in results if pool == "held-out")
    print("\nwrote %d read pairs" % written, file=sys.stderr)
    print("  %s" % o1, file=sys.stderr)
    print("  %s" % o2, file=sys.stderr)
    print("  %s  (per-read source pool and genome)" % truth, file=sys.stderr)
    print("  held-out genomes contributing: %d" % held, file=sys.stderr)


if __name__ == "__main__":
    main()
