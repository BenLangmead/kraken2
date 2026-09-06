# Notes for later

Ideas worked through but not built. Recorded so the reasoning does not have to
be redone.

## Databases larger than GPU memory

`pluspf_16_GB` fits an L40S with room to spare. Standard `pluspf` (~70 GB) and
`nt` do not. Three options, roughly in order of how much work they are.

### Use a size-capped database

The ecosystem already distributes them, sized for memory-constrained settings,
and they map onto VRAM exactly. This is the answer until something forces
otherwise.

### Spatial partitioning across several GPUs

`g6e.12xlarge` carries four L40S, 192 GB in total, which holds a 70 GB database
outright. Each GPU keeps one partition permanently, reads are broadcast, and the
results are reduced. No streaming, no slot management. Roughly four times the
hourly cost of a single-GPU box, and much the simpler engineering problem.

### Double-buffered partition streaming on one GPU

Two slots, each a large fraction of VRAM. The fused kernel runs against the
partition in one slot while the next partition loads into the other.

The loop has to be partition-outer:

    for each partition:              # streamed into alternating slots
        for each read batch:         # reads stay resident
            probe, accumulate

Reads-outer would reload every partition for every batch. Partition-outer pays
each transfer once, but needs per-read state that survives the whole partition
loop, and that state competes with the slots for the same memory.

Two things keep the state small. Minimizers should be recomputed per partition
rather than stored: storing costs about 312 bytes per read, recomputing costs
around 60 ms per pass at the measured 50.4 Gbase/s. And the accumulator should
be per-read counts rather than the per-token array, roughly 32 bytes against
156, which buys a five times larger read batch.

Rough shape for a 70 GB database in four partitions over 1B reads: about 32 s of
transfer against 26 s of compute, so transfer sets the floor and the overlap
hides everything else. Cost scales with partition count, so four partitions is
about four times a resident database. That assumed pinned transfers near
20 GB/s; the only figure actually measured is 12.8 GB/s with pageable memory, so
**this needs measuring before it is trusted** -- pageable would make it 50 s.

Slots at exactly half of VRAM leave nothing for reads and state; something
nearer 40/40/20 is realistic, and the split is a real tuning knob. The overlap
also needs a separate copy stream and pinned staging, or the transfer serialises
behind the kernel and buys nothing.

## How the partition is cut, and what the reduction must then do

The choice of cut decides what the reduction can be, and whether it is exact.

**By hash value.** Every minimizer lands in exactly one partition, so per token
at most one partition answers and the counts add without double counting. But
the *taxa* are spread across every partition -- a given organism is the value of
many keys, and those keys are scattered. So no partition sees a complete picture
of any taxon, and **resolution cannot be done per partition**. The counts have to
be merged first and resolved once, which is exact but keeps a counts map per read
alive across the whole partition loop.

**By taxonomy subtree.** Keys can be assigned by their stored taxon, and the
taxonomy nodes above a split have to be replicated into every partition that
needs them. Keys whose taxon sits above the split are the awkward case: they
belong to no single subtree and have to be duplicated or held in a shared
partition. The reduction here is a fold of the per-partition calls under lowest
common ancestor -- the same shape `merge` already uses across databases.

The second is cheaper than it looks. A partition's call can be resolved and
folded immediately, so the state that survives the loop is a single running
ancestor, four bytes per read rather than a counts map. That makes read batches
far larger and cuts the number of partition passes accordingly.

It is, however, an approximation. `ResolveTree` picks the taxon with the highest
root-to-leaf path score, and the maximum over merged counts is not generally the
same as folding per-partition maxima. So the two schemes trade exactness against
the number of passes:

| cut | reduction | exact? | state per read |
| --- | --- | --- | --- |
| hash value | merge counts, resolve once | yes | counts map, ~32 B |
| taxonomy subtree | fold calls under LCA | no, approximate | running LCA, 4 B |

Either way a partitioned index has a different false-positive pattern from a
monolithic one, so results will not match the single-database run exactly. That
is why the bar for this work was set at agreement on classification decisions
rather than byte equality, and under partitioning even that bar becomes
approximate.

## Smaller things

- Tree resolution is done by one thread per read while the rest of the block
  waits. It cost 27% of throughput when it went in, and is the most obvious
  remaining speed-up: share it across a warp, or resolve several reads per block.
- Host-to-device transfers use pageable memory in the measurement harnesses.
  Pinned staging should be worth roughly 1.5x on that leg.
- `revcom_version 0` and 40-bit cells are unimplemented, so databases built
  before Kraken 2.0.8 or with wider cells would produce wrong answers rather than
  an error. They should be refused explicitly.
