CXX ?= g++
KRAKEN2_SKIP_FOPENMP ?= -fopenmp
CXXFLAGS = $(KRAKEN2_SKIP_FOPENMP) -Wall -std=c++17 -O3 -fPIC -g
CXXFLAGS += -DLINEAR_PROBING

# Indexed BGZF input (-Z) needs an inflate implementation.  libdeflate is the
# default because it is roughly twice as fast as zlib at this; building with
# KRAKEN2_INFLATE_LIB=-lz and -DKRAKEN2_ZLIB_INFLATE uses zlib instead, which
# the tree already links for k2mask.
KRAKEN2_INFLATE_LIB ?= -ldeflate
CFLAGS = -Wall -std=c99 -O2 -g

.PHONY: all clean install

PROGS = estimate_capacity build_db classify dump_table k2surgery lookup_accession_numbers k2mask blast_to_fasta merge libtax.so

all: $(PROGS)

install: $(PROGS)
	cp $(PROGS) "$(KRAKEN2_DIR)/"

clean:
	rm -f *.o $(PROGS)

taxonomy.o: taxonomy.cc taxonomy.h mmap_file.h
hyperloglogplus.o: hyperloglogplus.cc hyperloglogplus.h
mmap_file.o: mmap_file.cc mmap_file.h
compact_hash.o: compact_hash.cc compact_hash.h kv_store.h mmap_file.h kraken2_data.h
blocked_table.o: blocked_table.cc blocked_table.h compact_hash.h kv_store.h mmap_file.h taxonomy.h
mmscanner.o: mmscanner.cc mmscanner.h
fast_reader.o: fast_reader.cc fast_reader.h bgzf_index.h
bgzf_index.o: bgzf_index.cc bgzf_index.h
k2_profile.o: k2_profile.cc k2_profile.h
omp_hack.o: omp_hack.cc omp_hack.h
reports.o: reports.cc reports.h kraken2_data.h
aa_translate.o: aa_translate.cc aa_translate.h
utilities.o: utilities.cc utilities.h

classify.o: classify.cc blocked_table.h fast_reader.h bgzf_index.h k2_profile.h kraken2_data.h kv_store.h taxonomy.h mmscanner.h compact_hash.h aa_translate.h reports.h utilities.h readcounts.h
dump_table.o: dump_table.cc compact_hash.h taxonomy.h mmscanner.h kraken2_data.h reports.h
estimate_capacity.o: estimate_capacity.cc kv_store.h mmscanner.h fast_reader.h utilities.h
build_db.o: build_db.cc blocked_table.h build_db.h fast_reader.h taxonomy.h mmscanner.h compact_hash.h kv_store.h kraken2_data.h utilities.h
lookup_accession_numbers.o: lookup_accession_numbers.cc mmap_file.h utilities.h
k2mask.o: k2mask.cc fast_reader.h
k2bgzf.o: k2bgzf.cc
k2mindump.o: k2mindump.cc mmscanner.h fast_reader.h kraken2_data.h
merge.o: merge.cc taxonomy.cc reports.cc mmap_file.cc utilities.cc hyperloglogplus.cc

blast_defline.o: blast_defline.c
blast_to_fasta.o: blast_to_fasta.c
blast_utils.o: blast_utils.c

build_db: build_db.o blocked_table.o mmap_file.o compact_hash.o fast_reader.o bgzf_index.o taxonomy.o mmscanner.o omp_hack.o utilities.o
	$(CXX) $(CXXFLAGS) -o $@ build_db.o blocked_table.o mmap_file.o compact_hash.o fast_reader.o bgzf_index.o taxonomy.o mmscanner.o omp_hack.o utilities.o $(LDFLAGS) $(KRAKEN2_INFLATE_LIB)

classify: classify.o blocked_table.o reports.o hyperloglogplus.o mmap_file.o compact_hash.o taxonomy.o mmscanner.o omp_hack.o aa_translate.o utilities.o fast_reader.o bgzf_index.o k2_profile.o
	$(CXX) $(CXXFLAGS) -o $@ classify.o blocked_table.o fast_reader.o bgzf_index.o k2_profile.o reports.o hyperloglogplus.o mmap_file.o compact_hash.o taxonomy.o mmscanner.o omp_hack.o aa_translate.o utilities.o $(LDFLAGS) $(KRAKEN2_INFLATE_LIB)

estimate_capacity: estimate_capacity.o fast_reader.o bgzf_index.o mmscanner.o omp_hack.o utilities.o
	$(CXX) $(CXXFLAGS) -o $@ estimate_capacity.o fast_reader.o bgzf_index.o mmscanner.o omp_hack.o utilities.o $(LDFLAGS) $(KRAKEN2_INFLATE_LIB)

k2surgery.o: k2surgery.cc blocked_table.h taxonomy.h kv_store.h kraken2_headers.h

k2surgery: k2surgery.o blocked_table.o mmap_file.o compact_hash.o taxonomy.o omp_hack.o utilities.o
	$(CXX) $(CXXFLAGS) -o $@ k2surgery.o blocked_table.o mmap_file.o compact_hash.o taxonomy.o omp_hack.o utilities.o $(LDFLAGS)

dump_table: dump_table.o mmap_file.o compact_hash.o omp_hack.o taxonomy.o reports.o hyperloglogplus.o
	$(CXX) $(CXXFLAGS) -o $@ dump_table.o mmap_file.o compact_hash.o omp_hack.o taxonomy.o reports.o hyperloglogplus.o $(LDFLAGS)

lookup_accession_numbers: lookup_accession_numbers.o mmap_file.o omp_hack.o utilities.o
	$(CXX) $(CXXFLAGS) -o $@ lookup_accession_numbers.o mmap_file.o omp_hack.o utilities.o $(LDFLAGS)

k2mask: k2mask.o fast_reader.o bgzf_index.o
	$(CXX) $(CXXFLAGS) -o $@ k2mask.o fast_reader.o bgzf_index.o $(LDFLAGS) $(KRAKEN2_INFLATE_LIB) -lz -pthread

# Converts FASTQ to the record-aligned BGZF that classify -Z reads.  Not in
# PROGS because it is a preparation tool rather than part of a classify run.
k2bgzf: k2bgzf.o
	$(CXX) $(CXXFLAGS) -o $@ k2bgzf.o $(LDFLAGS) $(KRAKEN2_INFLATE_LIB)

# Reference minimizer stream, used as the oracle when checking a device scan.
k2mindump: k2mindump.o mmscanner.o fast_reader.o bgzf_index.o omp_hack.o mmap_file.o compact_hash.o
	$(CXX) $(CXXFLAGS) -o $@ k2mindump.o mmscanner.o fast_reader.o bgzf_index.o omp_hack.o mmap_file.o compact_hash.o $(LDFLAGS) $(KRAKEN2_INFLATE_LIB)

merge: merge.o taxonomy.o reports.o mmap_file.o utilities.o hyperloglogplus.o omp_hack.o
	$(CXX) $(CXXFLAGS) -o $@ merge.o taxonomy.o reports.o mmap_file.o utilities.o hyperloglogplus.o omp_hack.o $(LDFLAGS) -lc

blast_to_fasta: blast_to_fasta.o blast_defline.o blast_utils.o
	$(CC) $(CFLAGS) -o  $@  blast_to_fasta.o blast_utils.o blast_defline.o $(LDFLAGS) -lm -lc

libtax.o: libtax.cc

libtax.so: taxonomy.o mmap_file.o libtax.o
	$(CXX) $(CXXFLAGS) -shared -o libtax.so taxonomy.o mmap_file.o libtax.o $(LDFLAGS)
