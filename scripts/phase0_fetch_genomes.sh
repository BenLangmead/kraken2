#!/usr/bin/env bash
# Fetch reference genomes for the simulated Phase 0 / Phase 3 read set.
#
# Two pools:
#   in-db/    RefSeq representative complete genomes (bacteria, archaea, viral).
#             These are what pluspf is built from, so reads from them classify.
#   held-out/ Metazoan and plant genomes, which pluspf does not contain
#             (its eukaryotes are human, protozoa and fungi only).  Reads from
#             these supply the deliberately-unclassifiable fraction.
#
# Selection is seeded, so re-running reproduces the same genome set.
set -euo pipefail

WORK="${WORK:-$HOME/kraken2-gpu-work}"
GEN="$WORK/genomes"
N_BACTERIA="${N_BACTERIA:-200}"
N_ARCHAEA="${N_ARCHAEA:-20}"
N_VIRAL="${N_VIRAL:-30}"
SEED="${SEED:-20260903}"
JOBS="${JOBS:-4}"

mkdir -p "$GEN/in-db" "$GEN/held-out" "$GEN/summaries"

# ---- pick accessions from the RefSeq assembly summaries -------------------
pick() {  # $1=group  $2=count
  local group="$1" count="$2"
  local sum="$GEN/summaries/${group}_assembly_summary.txt"
  if [ ! -s "$sum" ]; then
    echo "fetching $group assembly summary..." >&2
    curl -fsSL "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/${group}/assembly_summary.txt" -o "$sum"
  fi
  # field 5  refseq_category, 11 version_status, 12 assembly_level, 20 ftp_path
  awk -F'\t' -v OFS='\t' '
    !/^#/ && ($5=="reference genome" || $5=="representative genome") &&
    $11=="latest" && $12=="Complete Genome" && $20 ~ /^https/ { print $1, $20 }
  ' "$sum" | sort -u | python3 -c "
import sys, random
rows = [l.rstrip('\n') for l in sys.stdin if l.strip()]
random.Random($SEED + len('$group')).shuffle(rows)
for r in rows[:$count]:
    print(r)
"
}

URLS="$GEN/in-db/urls.txt"
: > "$URLS"
for spec in "bacteria $N_BACTERIA" "archaea $N_ARCHAEA" "viral $N_VIRAL"; do
  set -- $spec
  pick "$1" "$2" >> "$URLS"
done
echo "selected $(wc -l < "$URLS") in-db genomes"

# ---- download in-db genomes ----------------------------------------------
cut -f2 "$URLS" | while read -r ftp; do
  base="$(basename "$ftp")"
  echo "${ftp}/${base}_genomic.fna.gz"
done > "$GEN/in-db/download.txt"

echo "downloading in-db genomes ($JOBS parallel)..."
( cd "$GEN/in-db" && xargs -P "$JOBS" -n1 curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors -O < download.txt )

# ---- held-out genomes: metazoa and plant, absent from pluspf -------------
# pluspf's eukaryotes are human, protozoa and fungi only, so invertebrate and
# plant reference genomes supply the deliberately-unclassifiable fraction.
# Resolved from the assembly summaries rather than hardcoded paths, which go
# stale whenever an assembly is revised.
: > "$GEN/held-out/download.txt"
for group in invertebrate plant; do
  sum="$GEN/summaries/${group}_assembly_summary.txt"
  if [ ! -s "$sum" ]; then
    echo "fetching $group assembly summary..." >&2
    curl -fsSL "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/${group}/assembly_summary.txt" -o "$sum"
  fi
  awk -F'\t' '
    !/^#/ && $11=="latest" && $20 ~ /^https/ &&
    ($8=="Drosophila melanogaster" || $8=="Caenorhabditis elegans" ||
     $8=="Arabidopsis thaliana") {
      sub(/\/+$/, "", $20);            # ftp_path may carry a trailing slash
      n=split($20,a,"/"); print $20 "/" a[n] "_genomic.fna.gz"
    }
  ' "$sum" | head -2 >> "$GEN/held-out/download.txt"
done
sort -u -o "$GEN/held-out/download.txt" "$GEN/held-out/download.txt"
echo "held-out targets: $(wc -l < "$GEN/held-out/download.txt")"
echo "downloading held-out genomes..."
( cd "$GEN/held-out" && xargs -P 2 -n1 curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors -O < download.txt )

echo
echo "in-db:    $(ls "$GEN/in-db"/*.fna.gz 2>/dev/null | wc -l) genomes, $(du -sh "$GEN/in-db" | cut -f1)"
echo "held-out: $(ls "$GEN/held-out"/*.fna.gz 2>/dev/null | wc -l) genomes, $(du -sh "$GEN/held-out" | cut -f1)"
