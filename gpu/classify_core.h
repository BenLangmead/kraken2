// The parts of classification that the device kernel and the host emulation
// share, so the two cannot drift apart: Kraken 2's minimizer scanner run
// sequentially (used for mates that contain an ambiguous base), tree
// resolution, and the formatting of one output line.  Everything here
// compiles both as CUDA and as plain C++.

#ifndef KRAKEN2_GPU_CLASSIFY_CORE_H_
#define KRAKEN2_GPU_CLASSIFY_CORE_H_

#include <cstdint>
#include "table.h"

namespace k2c {

// Token values that are not taxa, as in classify.cc (TAXID_MAX - 2 and
// TAXID_MAX, narrowed to 32 bits).  Taxa are internal IDs, far below these.
static const uint32_t AMBIG = 0xFFFFFFFDu;
static const uint32_t MATE = 0xFFFFFFFFu;

K2HD uint64_t revcomp(uint64_t kmer, int n) {
  kmer = ((kmer & 0xCCCCCCCCCCCCCCCCULL) >> 2) | ((kmer & 0x3333333333333333ULL) << 2);
  kmer = ((kmer & 0xF0F0F0F0F0F0F0F0ULL) >> 4) | ((kmer & 0x0F0F0F0F0F0F0F0FULL) << 4);
  kmer = ((kmer & 0xFF00FF00FF00FF00ULL) >> 8) | ((kmer & 0x00FF00FF00FF00FFULL) << 8);
  kmer = ((kmer & 0xFFFF0000FFFF0000ULL) >> 16) | ((kmer & 0x0000FFFF0000FFFFULL) << 16);
  kmer = (kmer >> 32) | (kmer << 32);
  return ((~kmer) >> (64 - n * 2)) & ((1ULL << (n * 2)) - 1);
}

K2HD int base_code(char c) {
  switch (c) {
    case 'A': case 'a': return 0;
    case 'C': case 'c': return 1;
    case 'G': case 'g': return 2;
    case 'T': case 't': return 3;
    default: return -1;
  }
}

// MinimizerScanner (src/mmscanner.cc) for DNA, with its early exit and the
// `<=` test in is_ambiguous(), restated without std::vector.  next() returns
// one token per call exactly when NextMinimizer returns non-null, with the
// same minimizer and ambiguity.  The window queue holds at most k - l + 1
// entries; QCAP bounds that.
struct Scanner {
  static const int QCAP = 64;
  int k, l;
  uint64_t ssm, tog, lmask;
  const char *str; int len, pos;
  int loaded, qpos, qhead, qn;
  uint64_t lmer, last_ambig, last_min;
  uint64_t qc[QCAP]; int qp[QCAP];

  K2HD void init(int k_, int l_, uint64_t ssm_, uint64_t tog_) {
    k = k_; l = l_; ssm = ssm_;
    lmask = (l * 2 >= 64) ? ~0ULL : ((1ULL << (l * 2)) - 1);
    tog = tog_ & lmask;
  }
  K2HD void load(const char *s, int n) {
    str = s; len = n; pos = 0;
    if (n + 1 < l) pos = len;
    loaded = 0; qpos = 0; qhead = 0; qn = 0;
    lmer = 0; last_ambig = 0; last_min = ~0ULL;
  }
  K2HD bool is_ambiguous() const { return (k > l && qpos <= k - l) || last_ambig != 0; }
  K2HD uint64_t &qback_c() { return qc[(qhead + qn - 1) % QCAP]; }

  // Returns false when the read is exhausted; otherwise sets *m to the
  // current minimizer (stale while the scanner refills after an ambiguous
  // base, as on the CPU) and *amb to is_ambiguous().
  K2HD bool next(uint64_t *m, bool *amb) {
    if (pos >= len) return false;
    bool changed = false;
    while (!changed) {
      if (loaded == l) loaded--;
      while (loaded < l && pos < len) {
        loaded++;
        lmer <<= 2; last_ambig <<= 2;
        int c = base_code(str[pos++]);
        if (c < 0) {
          qn = 0; qhead = 0; qpos = 0; lmer = 0; loaded = 0;
          last_ambig |= 3;
        } else lmer |= (uint64_t) c;
        lmer &= lmask; last_ambig &= lmask;
        if (pos >= k && loaded < l) { *m = last_min; *amb = is_ambiguous(); return true; }
      }
      if (loaded < l) return false;
      uint64_t rc = revcomp(lmer, l);
      uint64_t canon = lmer < rc ? lmer : rc;
      if (ssm) canon &= ssm;
      uint64_t cand = canon ^ tog;
      if (k == l) { last_min = cand ^ tog; *m = last_min; *amb = is_ambiguous(); return true; }
      while (qn > 0 && qback_c() > cand) qn--;
      if (qn == 0 && qpos >= k - l) changed = true;
      qc[(qhead + qn) % QCAP] = cand; qp[(qhead + qn) % QCAP] = qpos; qn++;
      if (qp[qhead] < qpos - k + l) { qhead = (qhead + 1) % QCAP; qn--; changed = true; }
      if (qpos == k - l) changed = true;
      qpos++;
      if (pos >= k) break;
    }
    last_min = qc[qhead] ^ tog;
    *m = last_min; *amb = is_ambiguous();
    return true;
  }
};

K2HD bool is_ancestor(const uint64_t *parent, uint64_t a, uint64_t b) {
  if (!a || !b) return false;
  while (b > a) b = parent[b];
  return b == a;
}

K2HD uint64_t lca(const uint64_t *parent, uint64_t a, uint64_t b) {
  if (!a || !b) return a ? a : b;
  while (a != b) { if (a > b) a = parent[a]; else b = parent[b]; }
  return a;
}

// Kraken 2's tree resolution at confidence 0.  Ties fold with the lowest
// common ancestor, which does not depend on visiting order, so the result
// matches the CPU's walk over an unordered_map.
K2HD uint32_t resolve(const uint64_t *parent, const uint32_t *tax, const uint32_t *cnt, int n) {
  uint64_t max_taxon = 0; uint32_t max_score = 0;
  for (int i = 0; i < n; i++) {
    uint32_t score = 0;
    for (int j = 0; j < n; j++)
      if (is_ancestor(parent, tax[j], tax[i])) score += cnt[j];
    if (score > max_score) { max_score = score; max_taxon = tax[i]; }
    else if (score == max_score) max_taxon = lca(parent, max_taxon, tax[i]);
  }
  return (uint32_t) max_taxon;
}

// Distinct taxa and their counts over a token array, skipping zeros and
// markers, then the call.  At most 64 distinct taxa are kept, as before.
K2HD uint32_t call_of(const uint64_t *parent, const uint32_t *tok, int n) {
  uint32_t tx[64], ct[64]; int nd = 0;
  for (int j = 0; j < n; j++) {
    uint32_t t = tok[j];
    if (!t || t >= AMBIG) continue;
    int q = 0; while (q < nd && tx[q] != t) q++;
    if (q == nd) { if (nd < 64) { tx[nd] = t; ct[nd] = 1; nd++; } }
    else ct[q]++;
  }
  return resolve(parent, tx, ct, nd);
}

K2HD int put_u64(char *p, uint64_t v) {
  if (v == 0) { p[0] = '0'; return 1; }
  char t[24]; int n = 0;
  while (v) { t[n++] = (char) ('0' + (v % 10)); v /= 10; }
  for (int i = 0; i < n; i++) p[i] = t[n - 1 - i];
  return n;
}

// One line of Kraken output, as FinishClassification and AddHitlistString
// write it with default options.  For a pair, the name loses a trailing /1 or
// /2 and the length field is "len1|len2".  Returns the line's length; sets
// *over and stops the hit list if it would pass wend.
K2HD int format_line(char *w0, const char *wend, uint32_t call, const uint64_t *extid,
                     const char *hdr, uint32_t hlen, bool paired, uint32_t len1, uint32_t len2,
                     const uint32_t *tok, int n, bool *over) {
  char *w = w0;
  *w++ = call ? 'C' : 'U'; *w++ = '\t';
  uint32_t hn = hlen;
  if (paired && hn > 2 && hdr[hn - 2] == '/' && (hdr[hn - 1] == '1' || hdr[hn - 1] == '2')) hn -= 2;
  for (uint32_t i = 0; i < hn; i++) *w++ = hdr[i];
  *w++ = '\t';
  w += put_u64(w, call ? extid[call] : 0); *w++ = '\t';
  w += put_u64(w, len1);
  if (paired) { *w++ = '|'; w += put_u64(w, len2); }
  *w++ = '\t';
  if (n == 0) { *w++ = '0'; *w++ = ':'; *w++ = '0'; }
  else {
    uint32_t last = tok[0]; int run = 1;
    for (int j = 1; j <= n; j++) {
      bool end = (j == n);
      if (!end && tok[j] == last) { run++; continue; }
      if (w >= wend) { *over = true; break; }
      if (last == MATE) { *w++ = '|'; *w++ = ':'; *w++ = '|'; if (!end) *w++ = ' '; }
      else if (last == AMBIG) { *w++ = 'A'; *w++ = ':'; w += put_u64(w, (uint64_t) run); *w++ = ' '; }
      else {
        w += put_u64(w, last ? extid[last] : 0); *w++ = ':'; w += put_u64(w, (uint64_t) run);
        if (!end) *w++ = ' ';
      }
      if (!end) { last = tok[j]; run = 1; }
    }
  }
  *w++ = '\n';
  return (int) (w - w0);
}

}  // namespace k2c

#endif  // KRAKEN2_GPU_CLASSIFY_CORE_H_
