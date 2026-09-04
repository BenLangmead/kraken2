#!/usr/bin/env bash
# Build the Phase 0 ablation ladder: one classify binary per ablation level.
#
# Level-independent objects are compiled once; only classify.o and k2_profile.o
# are recompiled per level.  Binaries land in $OUTDIR as classify_L0 .. classify_L6.
#
#   CXX=g++-15 ./scripts/phase0_build.sh [outdir]
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
OUTDIR="${1:-$HOME/kraken2-gpu-work/bin}"
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--fopenmp -Wall -std=c++11 -O3 -fPIC -DLINEAR_PROBING}"
MAXLEVEL=6

mkdir -p "$OUTDIR"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

SHARED_SRCS="reports.cc hyperloglogplus.cc mmap_file.cc compact_hash.cc \
taxonomy.cc seqreader.cc mmscanner.cc omp_hack.cc aa_translate.cc utilities.cc \
libtax.cc fast_reader.cc"

echo "building level-independent objects..."
SHARED_OBJS=""
for f in $SHARED_SRCS; do
  o="$WORK/${f%.cc}.o"
  $CXX $CXXFLAGS -c "$SRC/$f" -o "$o" &
  SHARED_OBJS="$SHARED_OBJS $o"
done
wait

for lvl in $(seq 0 $MAXLEVEL); do
  echo "building L$lvl ..."
  $CXX $CXXFLAGS -DK2_ABLATE=$lvl -c "$SRC/classify.cc"   -o "$WORK/classify_$lvl.o"
  $CXX $CXXFLAGS -DK2_ABLATE=$lvl -c "$SRC/k2_profile.cc" -o "$WORK/k2_profile_$lvl.o"
  $CXX $CXXFLAGS -o "$OUTDIR/classify_L$lvl" \
      "$WORK/classify_$lvl.o" "$WORK/k2_profile_$lvl.o" $SHARED_OBJS
done

echo
echo "ablation ladder in $OUTDIR:"
ls -la "$OUTDIR"/classify_L*
