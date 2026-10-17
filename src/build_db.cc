/*
 * Copyright 2013-2023, Derrick Wood <dwood@cs.jhu.edu>
 *
 * This file is part of the Kraken 2 taxonomic sequence classification system.
 */

#include "build_db.h"
#include "kraken2_headers.h"
#include "taxonomy.h"
#include "mmscanner.h"
#include "compact_hash.h"
#include "blocked_table.h"
#include "kraken2_data.h"
#include "utilities.h"
#include <mutex>
#include <deque>
#include <condition_variable>
#include <atomic>

using std::string;
using std::map;
using std::cerr;
using std::endl;
using std::vector;
using std::istringstream;
using std::ifstream;
using std::ofstream;
using namespace kraken2;

// These will likely be overridden by wrapper script,
// just keeping sane defaults in case they aren't
#define DEFAULT_BLOCK_SIZE (10 * 1024 * 1024)  // 10 MB
#define DEFAULT_SUBBLOCK_SIZE (1024)

void ParseCommandLine(int argc, char **argv, Options &opts);
void usage(int exit_code = EX_USAGE);
void ReadIDToTaxonMap(map<string, taxid_t> &id_map, string &filename);
void GenerateTaxonomy(Options &opts, map<string, taxid_t> &id_map);
void BuildBlockedTable(const Taxonomy &taxonomy,
    const map<string, taxid_t> &ID_to_taxon_map, const Options &opts,
    size_t bits_for_taxid);

struct TaxonSeqPair {
  taxid_t taxon;
  string seq;
};

int main(int argc, char **argv) {
  Options opts;
  opts.spaced_seed_mask = DEFAULT_SPACED_SEED_MASK;
  opts.toggle_mask = DEFAULT_TOGGLE_MASK;
  opts.input_is_protein = false;
  opts.num_threads = 1;
  opts.block_size = DEFAULT_BLOCK_SIZE;
  opts.subblock_size = DEFAULT_SUBBLOCK_SIZE;
  opts.cht_cell_size = 32;
  opts.requested_bits_for_taxid = 0;
  opts.min_clear_hash_value = 0;
  opts.maximum_capacity = 0;
  opts.deterministic_build = true;
  opts.blocked_table = false;
  opts.blocked_partition_bits = -1;
  opts.blocked_load = 0.99;
  opts.blocked_remainder_bits = 0;
  opts.blocked_kept_partitions = 0;
  ParseCommandLine(argc, argv, opts);

  omp_set_num_threads( opts.num_threads );

  map<string, taxid_t> ID_to_taxon_map;

  ReadIDToTaxonMap(ID_to_taxon_map, opts.ID_to_taxon_map_filename);
  GenerateTaxonomy(opts, ID_to_taxon_map);

  std::cerr << "Taxonomy parsed and converted." << std::endl;

  Taxonomy taxonomy(opts.taxonomy_filename.c_str(), false);
  taxonomy.GenerateExternalToInternalIDMap();
  size_t bits_needed_for_value = 1;
  while ((1 << bits_needed_for_value) < (ssize_t) taxonomy.node_count())
    bits_needed_for_value++;
  if (opts.requested_bits_for_taxid > 0 &&
      bits_needed_for_value > opts.requested_bits_for_taxid)
    errx(EX_DATAERR, "more bits required for storing taxid");

  size_t bits_for_taxid = bits_needed_for_value;
  if (bits_for_taxid < opts.requested_bits_for_taxid)
    bits_for_taxid = opts.requested_bits_for_taxid;

  auto actual_capacity = opts.capacity;
  if (opts.maximum_capacity) {
    double frac = opts.maximum_capacity * 1.0 / opts.capacity;
    if (frac > 1)
      errx(EX_DATAERR, "maximum capacity larger than requested capacity");
    opts.min_clear_hash_value = (uint64_t) ((1 - frac) * UINT64_MAX);
    actual_capacity = opts.maximum_capacity;
  }

  if (opts.blocked_table) {
    BuildBlockedTable(taxonomy, ID_to_taxon_map, opts, bits_for_taxid);
  } else if (opts.cht_cell_size == 32) {
    build<CompactHashCell>(taxonomy, ID_to_taxon_map, opts, actual_capacity, bits_for_taxid);
  } else if (opts.cht_cell_size == 40) {
    build<CompactHashCell40>(taxonomy, ID_to_taxon_map, opts, actual_capacity, bits_for_taxid);
  } else {
    errx(EX_DATAERR, "Unsupported CHT cell size");
  }

  IndexOptions index_opts;
  index_opts.k = opts.k;
  index_opts.l = opts.l;
  index_opts.spaced_seed_mask = opts.spaced_seed_mask;
  index_opts.toggle_mask = opts.toggle_mask;
  index_opts.dna_db = ! opts.input_is_protein;
  index_opts.minimum_acceptable_hash_value = opts.min_clear_hash_value;
  index_opts.revcom_version = CURRENT_REVCOM_VERSION;
  index_opts.db_version = 0;
  index_opts.db_type = 0;
  ofstream opts_fs(opts.options_filename);
  opts_fs.write((char *) &index_opts, sizeof(index_opts));
  if (! opts_fs.good())
    errx(EX_OSERR, "Unable to write options file %s",
         opts.options_filename.c_str());
  opts_fs.close();
  std::cerr << " complete." << std::endl;

  return 0;
}

// This function exists to deal with NCBI's use of \x01 characters to denote
// the start of a new FASTA header in the same line (for non-redundant DBs).
// We return all sequence IDs in a header line, not just the first.
vector<string> ExtractNCBISequenceIDs(const char *header, size_t header_len) {
  vector<string> list;
  string current_str;
  bool in_id = true;

  for (size_t i = 0; i < header_len; ++i) {
    if (header[i] == 0x01) {
      // 0x01 starts new ID at next char
      if (! current_str.empty())
        list.push_back(current_str);
      current_str.clear();
      in_id = true;
    }
    else if (in_id && isspace(header[i])) {
      // spaces end ID
      if (! current_str.empty())
        list.push_back(current_str);
      current_str.clear();
      in_id = false;
    }
    else if (in_id) {
      // Build ID string char by char
      current_str.push_back(header[i]);
    }
  }
  if (! current_str.empty())
    list.push_back(current_str);
  return list;
}

void ReadIDToTaxonMap(map<string, taxid_t> &id_map, string &filename) {
  ifstream map_file(filename.c_str());
  if (! map_file.good())
    err(EX_NOINPUT, "unable to read from '%s'", filename.c_str());
  string line;

  while (getline(map_file, line)) {
    string seq_id;
    taxid_t taxid;
    istringstream iss(line);
    iss >> seq_id;
    iss >> taxid;
    if (taxid)
      id_map[seq_id] = taxid;
  }
}

void GenerateTaxonomy(Options &opts, map<string, taxid_t> &id_map) {
  NCBITaxonomy ncbi_taxonomy(
    opts.ncbi_taxonomy_directory + "/nodes.dmp",
    opts.ncbi_taxonomy_directory + "/names.dmp"
  );
  for (auto &kv_pair : id_map) {
    if (kv_pair.second != 0) {
      ncbi_taxonomy.MarkNode(kv_pair.second);
    }
  }
  ncbi_taxonomy.ConvertToKrakenTaxonomy(opts.taxonomy_filename.c_str());
}

void ParseCommandLine(int argc, char **argv, Options &opts) {
  int opt;
  long long sig;

  while ((opt = getopt(argc, argv, "?hB:b:c:FH:m:n:o:t:k:l:M:p:r:s:S:T:XC:YP:e:R:K:")) != -1) {
    switch (opt) {
      case 'h' : case '?' :
        usage(0);
        break;
      case 'B' :
        sig = atoll(optarg);
        if (sig < 1)
          errx(EX_USAGE, "must have positive block size");
        opts.block_size = sig;
        break;
      case 'b' :
        sig = atoll(optarg);
        if (sig < 1)
          errx(EX_USAGE, "must have positive subblock size");
        opts.subblock_size = sig;
        break;
      case 'r' :
        sig = atoll(optarg);
        if (sig < 0)
          errx(EX_USAGE, "can't have negative bit storage");
        if (sig > 31)
          errx(EX_USAGE, "can't have more than 31 bits of storage for taxid");
        opts.requested_bits_for_taxid = sig;
        break;
      case 'p' :
        opts.num_threads = atoll(optarg);
        if (opts.num_threads < 1)
          errx(EX_USAGE, "can't have negative number of threads");
        if (opts.num_threads > omp_get_max_threads())
          errx(EX_USAGE, "OMP only wants you to use %d threads", omp_get_max_threads());
        break;
      case 'H' :
        opts.hashtable_filename = optarg;
        break;
      case 'm' :
        opts.ID_to_taxon_map_filename = optarg;
        break;
      case 'n' :
        opts.ncbi_taxonomy_directory = optarg;
        break;
      case 'o' :
        opts.options_filename = optarg;
        break;
      case 't' :
        opts.taxonomy_filename = optarg;
        break;
      case 'S' :
        opts.spaced_seed_mask = strtol(optarg, nullptr, 2);
        break;
      case 'T' :
        opts.toggle_mask = strtol(optarg, nullptr, 2);
        break;
      case 'k' :
        sig = atoll(optarg);
        if (sig < 1)
          errx(EX_USAGE, "k must be positive integer");
        opts.k = sig;
        break;
      case 'l' :
        sig = atoll(optarg);
        if (sig < 1)
          errx(EX_USAGE, "l must be positive integer");
        if (sig > 31)
          errx(EX_USAGE, "l must be no more than 31");
        opts.l = sig;
        break;
      case 'c' :
        sig = atoll(optarg);
        if (sig < 1)
          errx(EX_USAGE, "capacity must be positive integer");
        opts.capacity = sig;
        break;
      case 'M' :
        sig = atoll(optarg);
        if (sig < 1)
          errx(EX_USAGE, "max capacity must be positive integer");
        opts.maximum_capacity = sig;
        break;
      case 'F' :
        opts.deterministic_build = false;
        break;
      case 'X' :
        opts.input_is_protein = true;
        break;
      case 'Y' :
        opts.blocked_table = true;
        break;
      case 'P' :
        sig = atoll(optarg);
        if (sig < 0 || sig > 16)
          errx(EX_USAGE, "partition bits must be between 0 and 16");
        opts.blocked_partition_bits = sig;
        break;
      case 'e' :
        opts.blocked_load = atof(optarg);
        if (opts.blocked_load <= 0 || opts.blocked_load >= 1)
          errx(EX_USAGE, "blocked table load must be between 0 and 1");
        break;
      case 'R' :
        sig = atoll(optarg);
        if (sig < 1 || sig > 46)
          errx(EX_USAGE, "blocked table remainder bits must be between 1 and 46");
        opts.blocked_remainder_bits = sig;
        break;
      case 'K' :
        sig = atoll(optarg);
        if (sig < 1)
          errx(EX_USAGE, "blocked table kept partitions must be positive");
        opts.blocked_kept_partitions = sig;
        break;
      case 'C':
        sig = atoll(optarg);
        if (sig != 32 && sig != 40)
          errx(EX_USAGE, "CHT cell size should be either 32 or 40 bits");
        opts.cht_cell_size = sig;
        break;
    }
  }

  if (opts.spaced_seed_mask != DEFAULT_SPACED_SEED_MASK)
    ExpandSpacedSeedMask(opts.spaced_seed_mask,
      opts.input_is_protein ? BITS_PER_CHAR_PRO : BITS_PER_CHAR_DNA);
  if (opts.hashtable_filename.empty() ||
      opts.ID_to_taxon_map_filename.empty() ||
      opts.ncbi_taxonomy_directory.empty() ||
      opts.options_filename.empty() ||
      opts.taxonomy_filename.empty())
  {
    cerr << "missing mandatory filename parameter" << endl;
    usage();
  }
  if (opts.k == 0 || opts.l == 0 || (opts.capacity == 0 && ! opts.blocked_table)) {
    cerr << "missing mandatory integer parameter" << endl;
    usage();
  }
  if (opts.k < opts.l) {
    cerr << "k cannot be less than l" << endl;
    usage();
  }
  if (opts.block_size < opts.subblock_size) {
    cerr << "block size cannot be less than subblock size\n";
    usage();
  }
  if (opts.blocked_table && opts.maximum_capacity) {
    cerr << "a blocked table cannot be subsampled with -M" << endl;
    usage();
  }
  if (opts.blocked_table && opts.input_is_protein) {
    cerr << "blocked tables support only nucleotide databases" << endl;
    usage();
  }
  if (opts.maximum_capacity > opts.capacity) {
    cerr << "maximum capacity option shouldn't specify larger capacity than normal" << endl;
    usage();
  }
}

void usage(int exit_code) {
  cerr << "Usage: build_db <options>\n"
       << "\n"
       << "Options (*mandatory):\n"
       << "* -H FILENAME   Kraken 2 hash table filename\n"
       << "* -m FILENAME   Sequence ID to taxon map filename\n"
       << "* -t FILENAME   Kraken 2 taxonomy filename\n"
       << "* -n DIR        NCBI taxonomy directory name\n"
       << "* -o FILENAME   Kraken 2 options filename\n"
       << "* -k INT        Set length of k-mers\n"
       << "* -l INT        Set length of minimizers\n"
       << "* -c INT        Set capacity of hash table\n"
       << "  -M INT        Set maximum capacity of hash table (MiniKraken)\n"
       << "  -S BITSTRING  Spaced seed mask\n"
       << "  -T BITSTRING  Minimizer toggle mask\n"
       << "  -X            Input seqs. are proteins\n"
       << "  -p INT        Number of threads\n"
       << "  -F            Use fast, nondeterministic building method\n"
       << "  -B INT        Read block size\n"
       << "  -b INT        Read subblock size\n"
       << "  -r INT        Bit storage requested for taxid\n"
       << "  -C INT        CHT cell size (default: 32)\n"
       << "  -Y            Build a blocked table instead of a compact hash table\n"
       << "                (-c is then not needed)\n"
       << "  -P INT        Blocked table partition bits (default: chosen from the\n"
       << "                key count, up to 12, keeping at least 2^17 keys per\n"
       << "                partition)\n"
       << "  -e FLOAT      Blocked table slot load (default: 0.99)\n"
       << "  -R INT        Blocked table remainder bits; below the exact width,\n"
       << "                keys are stored as fingerprints and lookups can return\n"
       << "                false positives (default: exact)\n"
       << "  -K INT        Blocked table: keep only the first INT partitions, a\n"
       << "                fraction INT / 2^P of minimizer hash space; lookups\n"
       << "                of the rest miss (default: all; needs -P)" << endl;
  exit(exit_code);
}

// Collects every (minimizer, taxon) pair of the library into a blocked
// table.  Thread 0 reads sequences one block at a time, splits them into
// overlapping subblocks, and queues the subblocks in units; the other threads
// scan units as they arrive.  There is no barrier between blocks, so a thread
// that stops to reduce a partition holds up only its own unit while reading
// and scanning go on.  The table is a function of the set of pairs alone, so
// the result does not depend on sequence order, scheduling or thread count.
void BuildBlockedTable(const Taxonomy &taxonomy,
    const map<string, taxid_t> &ID_to_taxon_map, const Options &opts,
    size_t bits_for_taxid)
{
  uint64_t lmer_mask = (1ULL << (opts.l * BITS_PER_CHAR_DNA)) - 1;
  uint64_t pext_mask = opts.spaced_seed_mask ?
      (opts.spaced_seed_mask & lmer_mask) : lmer_mask;
  BlockedTableBuilder builder(pext_mask, opts.blocked_partition_bits,
                              opts.blocked_load, taxonomy,
                              opts.blocked_remainder_bits,
                              opts.blocked_kept_partitions);
  int threads = opts.num_threads;
  vector<BlockedTableBuilder::Buffer *> buffers(threads);
  for (int t = 0; t < threads; t++)
    buffers[t] = new BlockedTableBuilder::Buffer(builder);

  // A block owns its reader, and so the buffer its items point into, until the
  // last of its units is scanned.
  struct Item { const char *seq; size_t len; taxid_t taxid; size_t start, finish; };
  struct Block {
    FastReader reader;
    vector<Item> items;
    std::atomic<size_t> units_left;
  };
  struct Unit { Block *block; size_t begin, end; };
  const size_t UNIT_ITEMS = 64;
  std::deque<Unit> queue;
  std::mutex queue_lock;
  std::condition_variable work_ready, space_ready;
  bool input_done = false;

  auto scan_unit = [&](const Unit &u, MinimizerScanner &scanner,
                       BlockedTableBuilder::Buffer &buf) {
    Block &b = *u.block;
    for (size_t n = u.begin; n < u.end; n++) {
      const Item &it = b.items[n];
      scanner.LoadSequence(it.seq, it.len, it.start, it.finish);
      uint64_t *minimizer_ptr;
      uint64_t last = UINT64_MAX;
      while ((minimizer_ptr = scanner.NextMinimizer())) {
        if (scanner.is_ambiguous())
          continue;
        uint64_t m = *minimizer_ptr;
        if (m == last)
          continue;
        last = m;
        if (opts.min_clear_hash_value &&
            MurmurHash3(m) < opts.min_clear_hash_value)
          continue;
        buf.Add(m, (uint32_t) it.taxid);
      }
    }
    if (--b.units_left == 0)
      delete u.block;
  };

  size_t processed_seq_ct = 0;
  size_t processed_ch_ct = 0;
  int fd = fileno(stdin);
  StreamCursor cursor;
  bool have_input = PrimeStream(fd, cursor);
  string fault;
  #pragma omp parallel num_threads(threads)
  {
    const int tid = omp_get_thread_num();
    const int team = omp_get_num_threads();
    BlockedTableBuilder::Buffer &buf = *buffers[tid];
    MinimizerScanner scanner(opts.k, opts.l, opts.spaced_seed_mask,
                             true, opts.toggle_mask);
    if (tid == 0) {
      // Keeps about four units per scanning thread queued, which bounds
      // the sequence held in memory to a few blocks.
      const size_t max_queued = 4 * (size_t) std::max(team - 1, 1);
      while (have_input) {
        Block *block = new Block();
        if (! block->reader.LoadBlock(fd, cursor, opts.block_size)) {
          delete block;
          break;
        }
        block->reader.Parse();
        if (fault.empty())
          fault = block->reader.fault();
        for (size_t j = 0; j < block->reader.size(); j++) {
          const SeqView &sequence = block->reader[j];
          auto all_sequence_ids = ExtractNCBISequenceIDs(sequence.header,
                                                         sequence.header_len);
          taxid_t taxid = 0;
          for (auto &seqid : all_sequence_ids) {
            if (ID_to_taxon_map.count(seqid) == 0 ||
                ID_to_taxon_map.at(seqid) == 0)
              continue;
            taxid = taxonomy.LowestCommonAncestor(
                taxid, taxonomy.GetInternalID(ID_to_taxon_map.at(seqid)));
          }
          if (! taxid)
            continue;
          size_t len = sequence.seq_len;
          // Consecutive subblocks overlap by k - 1 characters, so every
          // k-mer lies wholly inside at least one of them.
          for (size_t i = 0; i < len; i += opts.subblock_size) {
            size_t finish = std::min(len, i + opts.subblock_size + opts.k - 1);
            Item item = { sequence.seq, len, taxid, i, finish };
            block->items.push_back(item);
            if (finish == len)
              break;
          }
          processed_seq_ct++;
          processed_ch_ct += len;
        }
        const size_t n_items = block->items.size();
        const size_t n_units = (n_items + UNIT_ITEMS - 1) / UNIT_ITEMS;
        if (n_units == 0) {
          delete block;
          continue;
        }
        block->units_left = n_units;
        if (team == 1) {
          for (size_t i = 0; i < n_items; i += UNIT_ITEMS)
            scan_unit(Unit{block, i, std::min(n_items, i + UNIT_ITEMS)}, scanner, buf);
        }
        else {
          std::unique_lock<std::mutex> g(queue_lock);
          for (size_t i = 0; i < n_items; i += UNIT_ITEMS) {
            space_ready.wait(g, [&] { return queue.size() < max_queued; });
            queue.push_back(Unit{block, i, std::min(n_items, i + UNIT_ITEMS)});
            work_ready.notify_one();
          }
        }
        if (isatty(fileno(stderr))) {
          std::cerr << "\rProcessed " << processed_seq_ct << " sequences ("
                    << processed_ch_ct << " bp)...";
        }
      }
      std::lock_guard<std::mutex> g(queue_lock);
      input_done = true;
      work_ready.notify_all();
    }
    else {
      while (true) {
        Unit u;
        {
          std::unique_lock<std::mutex> g(queue_lock);
          work_ready.wait(g, [&] { return ! queue.empty() || input_done; });
          if (queue.empty())
            break;
          u = queue.front();
          queue.pop_front();
          space_ready.notify_one();
        }
        scan_unit(u, scanner, buf);
      }
    }
  }
  if (! cursor.error.empty())
    errx(EX_IOERR, "%s (standard input)", cursor.error.c_str());
  if (! fault.empty())
    warnx("%s", fault.c_str());
  if (isatty(fileno(stderr)))
    std::cerr << "\r";
  std::cerr << "Completed processing of " << processed_seq_ct << " sequences, "
            << processed_ch_ct << " bp" << std::endl;
  // Each buffer's last records go to their partitions, in parallel since an
  // append can reduce a partition.
  #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
  for (int t = 0; t < threads; t++)
    delete buffers[t];

  std::cerr << "Placing blocked table... " << std::flush;
  builder.Write(opts.hashtable_filename.c_str(), bits_for_taxid, threads);
}
