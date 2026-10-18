#!/usr/bin/env bash
# Builds mmscanner_test and checks MinimizerScanner against the queue-based
# reference copy on random sequences and on data/*.fa.
#
# Usage: tests/scanner/run.sh [WORK_DIR]
# Environment: CXX (default c++), TRIALS (default 20000), SEED (default 1)

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
WORK=${1:-$(mktemp -d "${TMPDIR:-/tmp}/k2scan.XXXXXX")}
mkdir -p "$WORK"
CXX=${CXX:-c++}
F="-Wall -std=c++17 -O2 -g -I$REPO/src -I$HERE"
$CXX $F -c -o "$WORK/mmscanner.o" "$REPO/src/mmscanner.cc"
$CXX $F -c -o "$WORK/mmscanner_orig.o" "$HERE/mmscanner_orig.cc"
$CXX $F -o "$WORK/mmscanner_test" "$HERE/mmscanner_test.cc" \
  "$WORK/mmscanner.o" "$WORK/mmscanner_orig.o"
"$WORK/mmscanner_test" random "${TRIALS:-20000}" "${SEED:-1}"
"$WORK/mmscanner_test" fasta "$REPO"/data/*.fa
