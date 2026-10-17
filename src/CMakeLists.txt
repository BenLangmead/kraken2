add_executable(build_db
        build_db.cc
        blocked_table.cc
        mmap_file.cc
        compact_hash.cc
        taxonomy.cc
        fast_reader.cc
        bgzf_index.cc
        mmscanner.cc
        omp_hack.cc
        utilities.cc)

add_executable(classify
        classify.cc
        blocked_table.cc
        fast_reader.cc
        bgzf_index.cc
        k2_profile.cc
        reports.cc
        mmap_file.cc
        compact_hash.cc
        taxonomy.cc
        mmscanner.cc
        omp_hack.cc
        aa_translate.cc
        utilities.cc
        hyperloglogplus.cc)

add_executable(estimate_capacity
        estimate_capacity.cc
        fast_reader.cc
        bgzf_index.cc
        mmscanner.cc
        omp_hack.cc
        utilities.cc)

add_executable(dump_table
        dump_table.cc
        mmap_file.cc
        compact_hash.cc
        omp_hack.cc
        taxonomy.cc
        reports.cc
        hyperloglogplus.cc)

add_executable(lookup_accession_numbers
        lookup_accession_numbers.cc
        mmap_file.cc
        omp_hack.cc
        utilities.cc)

add_executable(k2mask
        k2mask.cc
        fast_reader.cc
        bgzf_index.cc)

add_executable(blast_to_fasta
        blast_to_fasta.c
        blast_utils.c
        blast_defline.c)

add_library(tax SHARED
       libtax.cc
       taxonomy.cc
       mmap_file.cc
)

target_link_libraries(build_db ${INFLATE_LIBRARIES})
target_link_libraries(classify ${INFLATE_LIBRARIES})
target_link_libraries(estimate_capacity ${INFLATE_LIBRARIES})
target_link_libraries(k2mask ${INFLATE_LIBRARIES} ${ZLIB_LIBRARIES} Threads::Threads)
target_link_libraries(blast_to_fasta ${MATH})
