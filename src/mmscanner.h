/*
 * Copyright 2013-2023, Derrick Wood <dwood@cs.jhu.edu>
 *
 * This file is part of the Kraken 2 taxonomic sequence classification system.
 */

#ifndef KRAKEN2_MMSCANNER_H_
#define KRAKEN2_MMSCANNER_H_

#include "kraken2_headers.h"
#include "kv_store.h"
#include <algorithm>

namespace kraken2 {

const uint64_t DEFAULT_TOGGLE_MASK = 0xe37e28c4271b5a2dULL;
const uint64_t DEFAULT_SPACED_SEED_MASK = 0;
const int BITS_PER_CHAR_DNA = 2;
const int BITS_PER_CHAR_PRO = 4;
const int CURRENT_REVCOM_VERSION = 1;

struct MinimizerData {
  uint64_t candidate;
  ssize_t pos;
};

class MinimizerScanner {
  public:
  // Create scanner for seq over interval [start, finish)
  // Will report length l minimizers for all k-mers in the interval
  // Minimizers ordered by XORing candidates w/ toggle_mask
  //   toggle_mask == 0 implies lexicographical ordering
  MinimizerScanner(ssize_t k, ssize_t l,
                   uint64_t spaced_seed_mask = DEFAULT_SPACED_SEED_MASK,
                   bool dna_sequence = true,
                   uint64_t toggle_mask = DEFAULT_TOGGLE_MASK,
                   int revcom_version = CURRENT_REVCOM_VERSION);

  // Pointer form: avoids materializing a std::string per record, and matches
  // the flat buffer a device kernel would be handed.
  void LoadSequence(const char *seq, size_t len, size_t start = 0,
      size_t finish = SIZE_MAX);
  void LoadSequence(const std::string &seq, size_t start = 0,
      size_t finish = SIZE_MAX);

  uint64_t *NextMinimizer();
  // Return last minimizer, only valid if NextMinimizer last returned non-NULL
  uint64_t last_minimizer() const { return last_minimizer_; }
  ssize_t k() const { return k_; }
  ssize_t l() const { return l_; }
  bool is_dna() const { return dna_; }
  // Code for character c: 0 to 3 (DNA) or 0 to 15 (protein), or UINT8_MAX
  // for a character that makes an l-mer ambiguous.
  uint8_t lookup_code(uint8_t c) const { return lookup_table_[c]; }
  // Whether the k-mer of the last output has an ambiguous base, or starts
  // before the interval.
  bool is_ambiguous() const { return ambig_; }

  private:
  uint64_t reverse_complement(uint64_t kmer, uint8_t n);
  uint64_t canonical_representation(uint64_t kmer, uint8_t n);
  void set_lookup_table_character(char ch, uint8_t val);

  const char *str_;         // sequence bases, not NUL-terminated
  size_t str_len_;
  ssize_t k_;
  ssize_t l_;
  size_t str_pos_, start_, finish_;
  uint64_t spaced_seed_mask_;
  bool dna_;
  uint64_t toggle_mask_;
  uint64_t lmer_;
  uint64_t rc_lmer_;        // reverse complement of lmer_ (revcom_version 1)
  uint64_t lmer_mask_;
  uint64_t last_minimizer_;
  ssize_t run_;             // unambiguous bases since the last reset, capped at k
  // Candidates of the last k - l + 1 l-mers, all ones for an incomplete one.
  std::vector<uint64_t> window_;
  size_t window_pos_;
  bool ambig_;
  uint8_t lookup_table_[UINT8_MAX + 1];
  const int revcom_version_;
};

// NextMinimizer reads one base per loop iteration and returns at most one
// output per base.
//
// A base at offset t from the start of the interval produces an output when
// it ends a k-mer (t + 1 >= k).  A subblock that starts past position k - l
// also produces an output at each earlier base that completes an l-mer and
// lies at sequence position k - 1 or later.
//
// The value is the minimum candidate over the window of k - l + 1 l-mers
// ending at the base, where an l-mer that is incomplete or overlaps an
// ambiguous base counts as all ones.  If the l-mer ending at the base is
// itself incomplete, the value is left as that of the previous output
// (initially all ones), which callers skip because the output is ambiguous.
//
// The output is ambiguous unless all k bases of its k-mer are unambiguous and
// inside the interval.
inline uint64_t *MinimizerScanner::NextMinimizer() {
  const int bits_per_char = dna_ ? BITS_PER_CHAR_DNA : BITS_PER_CHAR_PRO;
  const ssize_t w = k_ - l_ + 1;
  while (str_pos_ < finish_) {
    size_t i = str_pos_++;
    auto code = lookup_table_[(uint8_t) str_[i]];
    if (code == UINT8_MAX) {
      // No l-mer that overlaps this base is complete, so clear the window.
      run_ = 0;
      std::fill(window_.begin(), window_.end(), ~(uint64_t) 0);
    }
    else {
      lmer_ = ((lmer_ << bits_per_char) | code) & lmer_mask_;
      if (dna_)
        rc_lmer_ = (rc_lmer_ >> 2) | ((uint64_t) (code ^ 3) << (2 * (l_ - 1)));
      if (run_ < k_)
        run_++;
    }
    bool complete = run_ >= l_;
    uint64_t candidate = ~(uint64_t) 0;
    if (complete) {
      uint64_t canonical_lmer = ! dna_ ? lmer_
          : revcom_version_ ? (lmer_ < rc_lmer_ ? lmer_ : rc_lmer_)
          : canonical_representation(lmer_, l_);
      if (spaced_seed_mask_)
        canonical_lmer &= spaced_seed_mask_;
      candidate = canonical_lmer ^ toggle_mask_;
    }
    // The window is a ring buffer of the last k - l + 1 candidates.
    window_[window_pos_] = candidate;
    if (++window_pos_ == (size_t) w)
      window_pos_ = 0;
    bool ends_kmer = i + 1 - start_ >= (size_t) k_;
    if (! ends_kmer && ! (complete && i + 1 >= (size_t) k_))
      continue;
    if (complete) {
      uint64_t m = window_[0];
      for (ssize_t j = 1; j < w; j++)
        m = window_[j] < m ? window_[j] : m;
      last_minimizer_ = m ^ toggle_mask_;
    }
    ambig_ = run_ < k_;
    return &last_minimizer_;
  }
  return nullptr;
}

}

#endif
