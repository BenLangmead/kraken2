# Device prototypes

Each of these was checked against the CPU implementation rather than assumed to
agree with it, and each records a measurement taken on an L40S (`g6e`, us-east-1).

| file | what it does | status |
| --- | --- | --- |
| `scan_kernel.cu` | minimizer scan, one block per read | exact on 200,000 reads; 50.4 Gbase/s |
| `probe_bench.cu` | random gather modelling the compact hash probe | 5,397 Mprobe/s vs 580 on 32 CPU threads |
| `gpu_inflate.cc` | nvCOMP gzip decode of record-aligned BGZF blocks | byte-exact vs the original FASTQ; 34.3 GB/s |
| `fused.cu` | scan, subsample and probe together, real 16 GB table | logic verified via `emul.cc`; not yet run on a device |
| `emul.cc` | host emulation of `fused.cu`, minus the CUDA parts | exact on 50,000 reads / 262,335 lookups |
| `kernels.cuh` | the device kernels, shared so the two programs cannot drift | &mdash; |
| `k2gpu.cu` | end to end: BGZF in, one call per read out | first end-to-end run |

`emul.cc` exists because device time is the scarce resource.  Running the
kernel's logic on the host first caught two defects that would otherwise have
cost a paid instance to find: the duplicate-minimizer and subsampling rejections
were sharing one flag array, which is wrong because a minimizer the filter drops
still counts as the previous reported one, and in the device version that same
sharing was a read/write race between threads.

The oracle both are checked against comes from `src/k2mindump.cc`.

## Building

    nvcc -O3 -std=c++14 -arch=sm_89 -o fused fused.cu
    g++  -O3 -std=c++11 -o emul emul.cc

`gpu_inflate.cc` additionally needs nvCOMP:

    g++ -O3 -std=c++14 -I<nvcomp>/include -L<nvcomp>/lib64 -lnvcomp -lcudart \
        -o gpu_inflate gpu_inflate.cc
