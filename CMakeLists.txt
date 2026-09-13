cmake_minimum_required(VERSION 3.18)
project(cis5650_stream_compaction_test LANGUAGES CUDA CXX)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++11 for host code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

add_subdirectory(stream_compaction)

include_directories(.)

set(headers
    "src/testing_helpers.hpp"
    )

set(sources
    "src/main.cpp"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${CMAKE_PROJECT_NAME} stream_compaction)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})

# Part 5 benchmark harness (reproducible measurements for the write-up)
add_executable(stream_compaction_bench
    "src/bench.cu"
    )

# Mirror the architecture selection of the stream_compaction library (see
# stream_compaction/CMakeLists.txt): `native` needs CMake >= 3.24.
if(CMAKE_VERSION VERSION_LESS 3.23.0)
    set(BENCH_CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS 3.24.0)
    set(BENCH_CUDA_ARCHITECTURES all-major)
else()
    set(BENCH_CUDA_ARCHITECTURES native)
endif()
set_target_properties(stream_compaction_bench PROPERTIES CUDA_ARCHITECTURES ${BENCH_CUDA_ARCHITECTURES})
target_link_libraries(stream_compaction_bench stream_compaction)
target_compile_options(stream_compaction_bench PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/Zc:preprocessor>")
