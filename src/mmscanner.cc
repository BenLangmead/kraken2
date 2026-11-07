/*
 * Copyright 2013-2023, Derrick Wood <dwood@cs.jhu.edu>
 *
 * This file is part of the Kraken 2 taxonomic sequence classification system.
 */

#include "mmscanner.h"
#include <cassert>

using std::string;
using std::vector;

namespace kraken2 {

void MinimizerScanner::set_lookup_table_character(char c, uint8_t val) {
  lookup_table_[(int) c] = val;
  lookup_table_[tolower(c)] = val;
}

MinimizerScanner::MinimizerScanner(ssize_t k, ssize_t l,
    uint64_t spaced_seed_mask, bool dna_sequence, uint64_t toggle_mask,
    int revcom_version)
    : str_(nullptr), str_len_(0), k_(k), l_(l), str_pos_(0), start_(0), finish_(0),
      spaced_seed_mask_(spaced_seed_mask), dna_(dna_sequence),
      toggle_mask_(toggle_mask), run_(0), window_(k - l + 1, ~(uint64_t) 0),
      window_pos_(0), ambig_(false), revcom_version_(revcom_version)
{
  if (l_ > (ssize_t) ((sizeof(uint64_t) * 8 - 1) / (dna_ ? BITS_PER_CHAR_DNA : BITS_PER_CHAR_PRO)))
    errx(EX_SOFTWARE, "l exceeds size limits for minimizer %s scanner",
         dna_ ? "nucleotide" : "protein");
  lmer_mask_ = 1;
  lmer_mask_ <<= (l_ * (dna_ ? BITS_PER_CHAR_DNA : BITS_PER_CHAR_PRO));
  lmer_mask_--;
  toggle_mask_ &= lmer_mask_;
  // LoadSequence sets the interval; no sequence is loaded yet.
  for (int i = 0; i < UINT8_MAX + 1; i++)
    lookup_table_[i] = UINT8_MAX;
  if (dna_) {
    set_lookup_table_character('A', 0x00);
    set_lookup_table_character('C', 0x01);
    set_lookup_table_character('G', 0x02);
    set_lookup_table_character('T', 0x03);
  }
  else {
    // Reduced alphabet uses 15-letter alphabet from AD Solis (2015),
    // in Proteins (doi:10.1002/prot.24936)
    // This means that the ambiguous codes B (N/D) and Z (E/Q) aren't
    // usable here because they span multiple groups.  Although J (I/L)
    // could be used because I & L are in the same group, we treat it as
    // we do B, Z, and X for consistency and to ease future changes to
    // the code base.

    // stop codons/rare amino acids
    set_lookup_table_character('*', 0x00);
    set_lookup_table_character('U', 0x00);
    set_lookup_table_character('O', 0x00);
    // alanine
    set_lookup_table_character('A', 0x01);
    // asparagine, glutamine, serine
    set_lookup_table_character('N', 0x02);
    set_lookup_table_character('Q', 0x02);
    set_lookup_table_character('S', 0x02);
    // cysteine
    set_lookup_table_character('C', 0x03);
    // aspartic acid, glutamic acid
    set_lookup_table_character('D', 0x04);
    set_lookup_table_character('E', 0x04);
    // phenylalanine
    set_lookup_table_character('F', 0x05);
    // glycine
    set_lookup_table_character('G', 0x06);
    // histidine
    set_lookup_table_character('H', 0x07);
    // isoleucine, leucine
    set_lookup_table_character('I', 0x08);
    set_lookup_table_character('L', 0x08);
    // lysine
    set_lookup_table_character('K', 0x09);
    // proline
    set_lookup_table_character('P', 0x0a);
    // arginine
    set_lookup_table_character('R', 0x0b);
    // methionine, valine
    set_lookup_table_character('M', 0x0c);
    set_lookup_table_character('V', 0x0c);
    // threonine
    set_lookup_table_character('T', 0x0d);
    // tryptophan
    set_lookup_table_character('W', 0x0e);
    // tyrosine
    set_lookup_table_character('Y', 0x0f);
  }
}

void MinimizerScanner::LoadSequence(const string &seq, size_t start, size_t finish) {
  LoadSequence(seq.data(), seq.size(), start, finish);
}

void MinimizerScanner::LoadSequence(const char *seq, size_t len, size_t start,
                                    size_t finish) {
  str_ = seq;
  str_len_ = len;
  start_ = start;
  finish_ = finish;
  str_pos_ = start_;
  if (finish_ > str_len_)
    finish_ = str_len_;
  if ((ssize_t) (finish_ - start_) + 1 < l_)  // Invalidate scanner if interval < 1 l-mer
    str_pos_ = finish_;
  last_minimizer_ = ~0;
  lmer_ = 0;
  rc_lmer_ = 0;
  run_ = 0;
  std::fill(window_.begin(), window_.end(), ~(uint64_t) 0);
  window_pos_ = 0;
  ambig_ = false;
}

// Adapted for 64-bit DNA use from public domain code at:
//   https://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel
uint64_t MinimizerScanner::reverse_complement(uint64_t kmer, uint8_t n) {
  // Reverse bits (leaving bit pairs - nucleotides - intact)
  // swap consecutive pairs
  kmer = ((kmer & 0xCCCCCCCCCCCCCCCCUL) >> 2)
       | ((kmer & 0x3333333333333333UL) << 2);
  // swap consecutive nibbles
  kmer = ((kmer & 0xF0F0F0F0F0F0F0F0UL) >> 4)
       | ((kmer & 0x0F0F0F0F0F0F0F0FUL) << 4);
  // swap consecutive bytes
  kmer = ((kmer & 0xFF00FF00FF00FF00UL) >> 8)
       | ((kmer & 0x00FF00FF00FF00FFUL) << 8);
  // swap consecutive byte pairs
  kmer = ((kmer & 0xFFFF0000FFFF0000UL) >> 16)
       | ((kmer & 0x0000FFFF0000FFFFUL) << 16);
  // swap halves of 64-bit word
  kmer = ( kmer >> 32 ) | ( kmer << 32);
  // Then complement
  if (revcom_version_ == 0)
    // This branch present to maintain backwards compatibility with old DBs
    return (~kmer) & ((1ull << (n * 2)) - 1);
  else
    return ((~kmer) >> (sizeof(kmer) * CHAR_BIT - n * 2)) & ((1ull << (n * 2)) - 1);
}

uint64_t MinimizerScanner::canonical_representation(uint64_t kmer, uint8_t n) {
  uint64_t revcom = reverse_complement(kmer, n);
  return kmer < revcom ? kmer : revcom;
}

} // end namespace

// using namespace kraken2;

// int main(void) {
//   MinimizerScanner mm(4, 3, DEFAULT_SPACED_SEED_MASK, true, 0);
//   // std::string sequence =
//   // "ACCAGCAGGGCGCACGCATCGCGCTGCCCGAGCCCTCGACCCCCTTTCTCTGTGAGCAGCAGCCCGATTGGCCGTTGAAGGCACCACAACAACCCGACTGGCCGCTGCCCAGGCCGGCGGCGCCAACCCGGCCTGCGGCGCGGCCCCAGAGCCTGCCTTATTTCTCCCCCTACCGACCATCTCCGCAGCATGCCAGCCGTCCTCTCGCGATAGGGGACCGCCACAATAGCAACGCCAATCACCTTCCCACCCCAGCCACCAGCAGCAGCAGCACGCTCTCAGATGACGCCCTCTGGCAGGCCCTCGGCGGCGGGCGCCGCCCCTCGTCGTCATCGGCGGCGACGAACAACAACAGCAGCAACAACAACAGGCTCAGCTACCACCCAGCAAACGGGGAAGCTCGCCGCCTGCATCCAGCTACTGCGACGGCCACGGCGAAGGCGAAGGCGACGGCAGAACACCAGGCCCATCCCGACTACCCGCATATGAGCCCCTATGACGACGACAACGCCGACGCCGACGCCGACGCCGATAACAACGGCGATGTGGGTGGCGGGACTGGTGTGGTGGGCCTGGGCGTCGTGATGGCGCCGCGCCCGGCGAGCATGTTGCCGGCGCGGTTGCGGACGCGTGATCGGGCTCCGCCGGCGCTGCCGCTGCTGCCTGCGACGACGCTGGCGTCGGTGTCGTTGGGGCCGACGTTGAGTGGCCCGTTTGTTGCGCAGTTTGAGTGAGCGGTGGTTTGTCTTGNNNNNNNNNNNNNNNNCTTGCTTGTCTAGAGGGGGTGTGGATTGGGAAGTGGTGGCGGGGAAGAAGGGCGGGCGTTGGTTGATGGGACAGGACGAAGGGCTGGGTTTCATTATGAACCAATTGTCCACGGTTGGGTAAGATATAGAGGAGCAAAGCGCATTGCATATGATACCCCTGGATAAATGAGAGCAGCAATGATCGACTGAATCCTTCTGATGTCGGTGTCCGGAGAAAGGGATCGGTTCTGAAC";
//   std::string sequence = "GTCATGCACGTTCAC";
//   mm.LoadSequence(sequence);
//   size_t mm_count = 0;
//   while (mm.NextMinimizer()) {
//     mm_count += 1;
//   }

//   std::cout << "minimizers found = " << mm_count << std::endl;

//   return 0;
// }
