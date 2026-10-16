# Device prototypes

Each of these was checked against the CPU implementation rather than assumed to
agree with it, and each records a measurement taken on an L40S (`g6e`, us-east-1).

| file | what it does | status |
| --- | --- | --- |
| `scan_kernel.cu` | minimizer scan, one block per read | exact on 200,000 reads; 50.4 Gbase/s |
| `probe_bench.cu` | random gather modelling the compact hash probe | 5,397 Mprobe/s vs 580 on 32 CPU threads |
| `gpu_inflate.cc` | nvCOMP gzip decode of record-aligned BGZF blocks | byte-exact vs the original FASTQ; 34.3 GB/s |
| `emul.cc` | host emulation of the fused kernel, minus the CUDA parts | every line identical to classify on 2M reads and 2M pairs, both tables |
| `kernels.cuh` | the device kernels, shared so the two programs cannot drift | &mdash; |
| `classify_core.h` | Kraken 2's scanner for mates with ambiguous bases, tree resolution, line formatting | shared by the kernel and `emul.cc` |
| `table.h` | the table as host and device see it: compact hash or blocked, chosen by magic number; blocked version 1 (internal taxon IDs in slots) or 2 (dense taxon codes, decoded through the code-to-taxon array), but not a subsampled one | shared by the kernels and `emul.cc` |
| `k2gpu.cu` | end to end: BGZF in (single-end, or interleaved pairs from `k2bgzf in1.fq in2.fq`), one line per read or pair out; can stream a blocked table in groups of partitions | byte-identical to classify with either table, single-end or paired, resident or streamed (10M reads or pairs, L40S) |

`emul.cc` exists because device time is the scarce resource.  Running the
kernel's logic on the host first caught two defects that would otherwise have
cost a paid instance to find: the duplicate-minimizer and subsampling rejections
were sharing one flag array, which is wrong because a minimizer the filter drops
still counts as the previous reported one, and in the device version that same
sharing was a read/write race between threads.

Both `emul` and `k2gpu` are checked against the output of `classify` itself.

## Building

    g++  -O3 -std=c++11 -o emul emul.cc

`gpu_inflate.cc` and `k2gpu.cu` additionally need nvCOMP.  Use 5.3 or later:
5.0.0.6 reads outside its output buffer when inflating these BGZF blocks.

    g++ -O3 -std=c++14 -I<nvcomp>/include -L<nvcomp>/lib64 -lnvcomp -lcudart \
        -o gpu_inflate gpu_inflate.cc
