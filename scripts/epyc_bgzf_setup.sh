#!/usr/bin/env bash
# Provision one instance and run the indexed BGZF measurement.
#
# Expects the source tree to already be present at $SRC (scp'd, so nothing has
# to be pushed anywhere).  Everything else is fetched or built here.
set -euo pipefail
set -x

SRC=${SRC:-/data/kraken2}
WORK=${WORK:-/data}
DB=$WORK/db
READS=$WORK/reads
BIN=$WORK/bin
RESULTS=$WORK/results
THREADS=${THREADS:-8,16,32}
REPS=${REPS:-5}
NPROC=$(nproc)

sudo mkdir -p "$WORK" && sudo chown "$(id -u):$(id -g)" "$WORK"
mkdir -p "$DB" "$READS" "$BIN" "$RESULTS"

# ---- dependencies -----------------------------------------------------------
if command -v apt-get > /dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq build-essential python3 libdeflate-dev unzip curl
elif command -v dnf > /dev/null; then
  sudo dnf install -y -q gcc-c++ make python3 libdeflate-devel unzip tar
fi
# Fall back to a source build if the distro has no libdeflate package.
if ! echo '#include <libdeflate.h>' | cc -E - > /dev/null 2>&1; then
  curl -sL https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.19.tar.gz | tar xz -C /tmp
  make -C /tmp/libdeflate-1.19 -j"$NPROC" > /dev/null
  sudo make -C /tmp/libdeflate-1.19 install > /dev/null
  sudo ldconfig
fi
if ! command -v aws > /dev/null; then
  curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip
  unzip -q /tmp/awscli.zip -d /tmp && sudo /tmp/aws/install > /dev/null
fi

# ---- build ------------------------------------------------------------------
CF="-fopenmp -Wall -std=c++11 -O3 -fPIC -g -DLINEAR_PROBING"
make -C "$SRC/src" clean > /dev/null
make -C "$SRC/src" -j"$NPROC" CXXFLAGS="$CF -DK2_ABLATE=6" classify > /dev/null
cp "$SRC/src/classify" "$BIN/classify_L6"
make -C "$SRC/src" clean > /dev/null
make -C "$SRC/src" -j"$NPROC" CXXFLAGS="$CF" classify k2bgzf > /dev/null
cp "$SRC/src/classify" "$BIN/classify_L0"
cp "$SRC/src/k2bgzf" "$BIN/k2bgzf"

# ---- inputs -----------------------------------------------------------------
B=https://genome-idx.s3.amazonaws.com/kraken
for f in hash.k2d taxo.k2d opts.k2d; do
  [ -s "$DB/$f" ] || curl -sSL -o "$DB/$f" "$B/pluspf_16_GB_20260626/$f"
done
for f in sim10m_1.fq.gz sim10m_2.fq.gz; do
  [ -s "$READS/$f" ] || curl -sSL -o "$READS/$f" "$B/bench/sim26/$f"
done
for i in 1 2; do
  [ -s "$READS/sim10m_$i.fq" ] || gzip -dc "$READS/sim10m_$i.fq.gz" > "$READS/sim10m_$i.fq"
done

# ---- record-aligned BGZF ----------------------------------------------------
for i in 1 2; do
  [ -s "$READS/sim10m_$i.k2bgz" ] || \
    "$BIN/k2bgzf" -t "$NPROC" -o "$READS/sim10m_$i.k2bgz" \
                  -i "$READS/sim10m_$i.k2i" "$READS/sim10m_$i.fq"
done
# The container must still be an ordinary gzip file, and must round trip.
gzip -dc "$READS/sim10m_1.k2bgz" | cmp - "$READS/sim10m_1.fq"
gzip -dc "$READS/sim10m_2.k2bgz" | cmp - "$READS/sim10m_2.fq"

df -h "$WORK"; free -g; lscpu | head -20

# ---- measure ----------------------------------------------------------------
python3 -u "$SRC/scripts/bgzf_bench.py" \
  --bin-dir "$BIN" --db "$DB" \
  --plain "$READS/sim10m_1.fq"    "$READS/sim10m_2.fq" \
  --bgzf  "$READS/sim10m_1.k2bgz" "$READS/sim10m_2.k2bgz" \
  --gz    "$READS/sim10m_1.fq.gz" "$READS/sim10m_2.fq.gz" \
  --threads "$THREADS" --reps "$REPS" --out "$RESULTS" --label epyc_bgzf

touch "$WORK/DONE"
