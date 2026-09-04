#!/usr/bin/env python3
"""Measure indexed BGZF input against plain and gzipped FASTQ.

The question is narrow.  classify's input critical section holds a serial region
whose cost does not fall with thread count, and on a 32 core EPYC that region is
most of the runtime.  Indexed BGZF removes the region but adds inflate work that
competes with classification.  Which wins depends on the machine, and a 12 core
laptop cannot reach the regime where the region dominates, so this has to be
measured where it does.

Three arms supply identical reads three ways: plain FASTQ, gzip through a pipe
as the kraken2 wrapper does it, and record-aligned BGZF read with -Z.  Two
levels: the full pipeline, and K2_ABLATE=6, which strips scanning and probing so
only the input path remains.  The second is what isolates the region under test;
the first is what anyone actually cares about.

Ordering is shuffled within each repetition rather than run arm by arm, because
a long session drifts thermally and level-major ordering lets that drift
masquerade as an effect.  Equivalence is checked once per arm before any timing:
a speedup measured against different output would not mean anything.
"""

import argparse, csv, json, os, random, re, shlex, shutil, statistics, subprocess, sys, tempfile, time

TIME_RE = re.compile(r"processed in ([0-9.]+)s")


# A timeout matters more than it looks.  The gzip arm feeds classify through
# FIFOs, so if classify exits early the writers block on a full pipe and the
# shell's wait never returns.  Unattended on a billed instance that is an
# unbounded hang, not a failed cell.
def run(cmd, capture=True, timeout=1800):
    try:
        p = subprocess.run(cmd, shell=True,
                           stdout=subprocess.PIPE if capture else None,
                           stderr=subprocess.STDOUT, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout}s"
    return p.returncode, (p.stdout or "")


def classify_cmd(binary, db, arm, reads, threads, report, out, paired):
    base = (f"{shlex.quote(binary)} -H {db}/hash.k2d -t {db}/taxo.k2d "
            f"-o {db}/opts.k2d -M -p {threads} "
            f"-R {shlex.quote(report)} -O {shlex.quote(out)}")
    if paired:
        base += " -P"
    if arm == "bgzf":
        return f"{base} -Z {' '.join(shlex.quote(r) for r in reads)}"
    if arm == "plain":
        return f"{base} {' '.join(shlex.quote(r) for r in reads)}"
    if arm == "gzip":
        # The kraken2 wrapper feeds classify through one decompressor per file.
        # Reproduce that, including the FIFOs it uses for a pair.
        if paired:
            d = tempfile.mkdtemp(prefix="k2gz")
            f1, f2 = os.path.join(d, "m1"), os.path.join(d, "m2")
            os.mkfifo(f1); os.mkfifo(f2)
            # rm -rf runs whatever happens, so a failed cell does not leave a
            # FIFO behind for every repetition.
            return (f"( gzip -dc {shlex.quote(reads[0])} > {f1} & "
                    f"gzip -dc {shlex.quote(reads[1])} > {f2} & "
                    f"{base} {f1} {f2}; wait ); rm -rf {shlex.quote(d)}")
        return f"gzip -dc {shlex.quote(reads[0])} | {base}"
    raise ValueError(arm)


def timed(cmd):
    t0 = time.time()
    rc, out = run(cmd)
    wall = time.time() - t0
    m = TIME_RE.search(out)
    if rc != 0 or not m:
        return None, wall, out.strip()[-500:]
    return float(m.group(1)), wall, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True,
                    help="holds classify_L0 and classify_L6")
    ap.add_argument("--db", required=True)
    ap.add_argument("--plain", nargs="+", required=True)
    ap.add_argument("--bgzf", nargs="+", required=True)
    ap.add_argument("--gz", nargs="+", required=True)
    ap.add_argument("--threads", default="8,16,32")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="bgzf")
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    threads = [int(t) for t in args.threads.split(",")]
    paired = len(args.plain) == 2
    arms = {"plain": args.plain, "gzip": args.gz, "bgzf": args.bgzf}
    os.makedirs(args.out, exist_ok=True)
    L0 = os.path.join(args.bin_dir, "classify_L0")
    L6 = os.path.join(args.bin_dir, "classify_L6")

    # ---- equivalence, once, before anything is timed -----------------------
    print("checking equivalence across arms ...", flush=True)
    ref_out = ref_rep = None
    for arm in ("plain", "gzip", "bgzf"):
        o = os.path.join(args.out, f"{args.label}_{arm}.out")
        r = os.path.join(args.out, f"{args.label}_{arm}.report")
        cmd = classify_cmd(L0, args.db, arm, arms[arm], 8, r, o, paired)
        rc, log = run(cmd)
        if rc != 0:
            print(f"  {arm}: FAILED\n{log[-800:]}", flush=True)
            return 2
        if ref_out is None:
            ref_out, ref_rep = o, r
            print(f"  {arm}: reference", flush=True)
        else:
            same_o = subprocess.run(["cmp", "-s", ref_out, o]).returncode == 0
            same_r = subprocess.run(["cmp", "-s", ref_rep, r]).returncode == 0
            print(f"  {arm}: output {'OK' if same_o else 'DIFFERS'}, "
                  f"report {'OK' if same_r else 'DIFFERS'}", flush=True)
            if not (same_o and same_r):
                print("  refusing to time arms that disagree", flush=True)
                return 3

    # ---- warm the page cache, discarded ------------------------------------
    print("warming ...", flush=True)
    for arm in arms:
        timed(classify_cmd(L0, args.db, arm, arms[arm], max(threads),
                           "/dev/null", "-", paired))

    # ---- timed cells, shuffled within each repetition ----------------------
    cells = [(lvl, arm, t) for lvl in ("L0", "L6")
             for arm in arms for t in threads]
    rng = random.Random(args.seed)
    raw = []
    for rep in range(1, args.reps + 1):
        order = cells[:]
        rng.shuffle(order)
        for lvl, arm, t in order:
            binary = L0 if lvl == "L0" else L6
            secs, wall, err = timed(classify_cmd(binary, args.db, arm, arms[arm],
                                                 t, "/dev/null", "-", paired))
            if secs is None:
                print(f"  rep {rep} {lvl} {arm:<5} p={t:<3} FAILED: {err}", flush=True)
                continue
            raw.append(dict(rep=rep, level=lvl, arm=arm, threads=t,
                            seconds=secs, wall=wall))
            print(f"  rep {rep} {lvl} {arm:<5} p={t:<3} {secs:7.3f}s", flush=True)

    rawpath = os.path.join(args.out, f"{args.label}_raw.csv")
    with open(rawpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rep", "level", "arm", "threads",
                                          "seconds", "wall"])
        w.writeheader()
        w.writerows(raw)

    # ---- summary ------------------------------------------------------------
    def best(lvl, arm, t):
        v = [r["seconds"] for r in raw
             if r["level"] == lvl and r["arm"] == arm and r["threads"] == t]
        return (min(v), statistics.median(v), (max(v) - min(v)) / min(v) * 100) if v else None

    lines = []
    for lvl in ("L0", "L6"):
        lines.append(f"\n{'full pipeline' if lvl == 'L0' else 'input path only (K2_ABLATE=6)'}")
        lines.append(f"{'p':>4}  {'plain':>18}  {'gzip':>18}  {'bgzf':>18}   "
                     f"{'bgzf/plain':>10} {'bgzf/gzip':>10}")
        for t in threads:
            row = {a: best(lvl, a, t) for a in arms}
            if any(v is None for v in row.values()):
                continue
            fmt = lambda v: f"{v[0]:7.3f}s ~{v[2]:4.1f}%"
            lines.append(f"{t:>4}  {fmt(row['plain']):>18}  {fmt(row['gzip']):>18}  "
                         f"{fmt(row['bgzf']):>18}   "
                         f"{row['plain'][0]/row['bgzf'][0]:9.2f}x "
                         f"{row['gzip'][0]/row['bgzf'][0]:9.2f}x")
    summary = "\n".join(lines)
    print(summary, flush=True)
    with open(os.path.join(args.out, f"{args.label}_summary.txt"), "w") as f:
        f.write(summary + "\n")
    with open(os.path.join(args.out, f"{args.label}_raw.json"), "w") as f:
        json.dump(raw, f, indent=1)
    print(f"\nwrote {rawpath}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
