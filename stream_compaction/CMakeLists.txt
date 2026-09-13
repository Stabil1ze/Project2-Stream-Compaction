set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "radix_sort.h"
    "shared_scan.h"
    "thrust.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "radix_sort.cu"
    "shared_scan.cu"
    "thrust.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_library(stream_compaction ${sources} ${headers})
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES all-major)
else()
set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES native)
endif()
target_compile_options(stream_compaction PRIVATE "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<PLATFORM_ID:Windows>>:-Xcompiler=/Zc:preprocessor>")
target_compile_options(stream_compaction PRIVATE "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
target_compile_options(stream_compaction PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")
