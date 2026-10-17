cmake_minimum_required(VERSION 2.8...3.11)

project(kraken2)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall)
add_definitions(-DLINEAR_PROBING)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
    message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)

find_package(Threads REQUIRED)
find_library(MATH m REQUIRED)
find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

# Indexed BGZF input (classify -Z) inflates with libdeflate when it is
# installed, and with zlib otherwise.
find_library(DEFLATE_LIBRARY deflate)
find_path(DEFLATE_INCLUDE_DIR libdeflate.h)
if(DEFLATE_LIBRARY AND DEFLATE_INCLUDE_DIR)
    include_directories(${DEFLATE_INCLUDE_DIR})
    set(INFLATE_LIBRARIES ${DEFLATE_LIBRARY})
else()
    add_definitions(-DKRAKEN2_ZLIB_INFLATE)
    set(INFLATE_LIBRARIES ${ZLIB_LIBRARIES})
endif()

add_subdirectory(src)
