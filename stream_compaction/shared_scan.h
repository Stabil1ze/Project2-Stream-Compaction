#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedScan {

        StreamCompaction::Common::PerformanceTimer& timer();

        const int DEFAULT_BLOCK_THREADS = 256;

        void scanNaive(int n, int *odata, const int *idata,
                       int blockThreads = DEFAULT_BLOCK_THREADS);

        void scanEfficient(int n, int *odata, const int *idata,
                           int blockThreads = DEFAULT_BLOCK_THREADS);

        void scanEfficientPadded(int n, int *odata, const int *idata,
                                 int blockThreads = DEFAULT_BLOCK_THREADS);
    }
}
