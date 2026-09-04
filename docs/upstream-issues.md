# Upstream issues found while rebasing onto DerrickWood/kraken2

Notes on problems observed in `upstream/master` (400b52d, "Fix additional bugs in
merge.cc"). Recorded here so they can be turned into issues or PRs later; none of
them are caused by our changes.

## 1. `TOK_REPEAT` feeds the wrong k-mer into the HyperLogLog sketch, ungated

`src/classify.cc`, phase 3 of `ClassifySequence`:

```cpp
        case TOK_LOOKUP:
          taxon = lookup_vals[tok.key_idx];
          last_taxon = taxon;
          if (taxon) {
            minimizer_hit_groups++;
            if (!opts.report_filename.empty() || !opts.taxon_counters_dump_filename.empty())
              curr_taxon_counts[taxon].add_kmer(lookup_keys[tok.key_idx]);
          }
          break;
        default:  // TOK_REPEAT
          taxon = last_taxon;
          curr_taxon_counts[taxon].add_kmer(lookup_keys[tok.key_idx]);
          break;
```

Four separate problems in that one `TOK_REPEAT` line:

1. **Wrong k-mer.** Repeat tokens are pushed as `{TOK_REPEAT, 0}`, so `tok.key_idx`
   is always 0 and the call always adds `lookup_keys[0]`, not the minimizer the
   token actually stands for.
2. **Not gated.** The `TOK_LOOKUP` branch immediately above only registers a
   k-mer when a report or a counters dump was requested. The repeat branch does it
   unconditionally.
3. **`taxon` may be zero.** After a miss, `last_taxon` is 0, so this creates and
   populates a counter for taxon 0.
4. **Out-of-bounds when nothing was looked up.** If every distinct minimizer in a
   fragment was skipped by `minimum_acceptable_hash_value`, `lookup_keys` is empty
   and `lookup_keys[0]` is out of range. With a capped database such as
   `pluspf_16_GB`, whose `minimum_acceptable_hash_value` skips about 86% of
   minimizers, that is not a rare shape.

AddressSanitizer does not flag (4): `lookup_keys` is `static thread_local` and is
`clear()`ed rather than freed, so after the first fragment the read lands inside a
still-live allocation and silently returns a stale key.

Semantically, the pre-refactor code only registered a k-mer when the minimizer
*changed*, never on a repeat, so this also inflates distinct-k-mer estimates.

Our batched path does not reproduce this, so under `-K` our report's clade k-mer
and distinct-k-mer columns differ from upstream's; every other column, including
all read counts, matches. We do keep one inert side effect of the bug: it creates
a counter entry for taxon 0, and because `KrakenReportDFS` orders sibling taxa
with a comparator that branches on whether a taxon is present in the map at all,
dropping that entry reorders equal-count rows. We create it once per block so
reports stay byte-identical, and that line should go when this is fixed.

Cost is small: removing the line changed 10M read pairs from 25.11s to 24.76s at 8
threads, so this is a correctness report rather than a performance one.

## 2. Removing the per-position return from `NextMinimizer` changed the meaning of the hitlist and of `--confidence`

`src/mmscanner.cc`, in `NextMinimizer`. `git blame` attributes the commenting-out
to `720f3718`, Rone Charles, 2026-08-27, "Add support for report-minimizer-data
when doing multi-database classification" (only the comment line above it is
older, from Derrick Wood in 2020):

```cpp
    // Return only if we've read in at least one k-mer's worth of chars
    // if (str_pos_ >= (size_t) k_) {
    //   break;
    // }
```

With that `break` present the scanner returns once per k-mer position; without it
the enclosing `while (! changed_minimizer)` loop runs on, so it returns once per
*distinct minimizer*. `classify.cc` still does one `taxa.push_back(taxon)` and one
`hit_counts[taxon]++` per call, so the change silently redefines all three of:

- **The hitlist column.** Run counts become per minimizer group rather than per
  k-mer. For one 150 bp mate at k=35 the counts sum to 37 rather than 116.
- **Taxon weighting in `ResolveTree`.** A minimizer covering a 60-position run used
  to contribute 60 to `hit_counts` and now contributes 1, so a taxon supported by
  one long run no longer outweighs one supported by two short runs.
- **The `--confidence` denominator**, which is `total_kmers = taxa.size()`.

Measured effect on 100k simulated pairs against `pluspf_16_GB`:

| `-T` | before (2.1.1) | after |
| ---: | ---: | ---: |
| 0    | 75.90% | 75.90% |
| 0.05 | 71.59% | 72.76% |
| 0.1  | 50.46% | 53.56% |
| 0.2  | 4.04%  | 2.96%  |

Identical at `-T 0`, as expected, and diverging by about 3 percentage points at
`-T 0.1`. The Kraken 2 paper defines the confidence score over k-mers, which is the
older behavior, and nothing in the changelog mentions the change.

The commit it comes from is about minimizer-data reporting, so returning once per
distinct minimizer may well be deliberate for that purpose; what appears
unintended is that the same counter drives the hitlist and the `--confidence`
denominator. The commit also carries work-in-progress markers: the `break` is
commented out rather than removed, and a commented-out `main()` is left at the
bottom of the file containing a debug sequence and a loop that counts how many
minimizers the scanner emits. It bundles a real fix too, `lmer_ = 0` in
`LoadSequence`, which had been leaking l-mer bits across sequences. At the time of
writing the commit is eight days old, so this may not be in a release yet, and
asking what was intended is more useful than filing it as a defect.
