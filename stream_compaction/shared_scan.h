#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace SharedScan {

        StreamCompaction::Common::PerformanceTimer& timer();

        // Default tile size (one element per thread). Must be a power of two; the
        // host code clamps it into [32, 1024] and rounds it up if needed.
        const int DEFAULT_BLOCK_THREADS = 256;

        /**
         * Exclusive scan with GPU Gems 3, Chapter 39, Example 39-1: a Hillis-Steele
         * scan held in shared memory, one block per tile of `blockThreads`
         * elements. The tile totals are scanned in a second pass and added back in
         * a third, which extends the block scan to arrays of arbitrary size.
         */
        void scanNaive(int n, int *odata, const int *idata,
                       int blockThreads = DEFAULT_BLOCK_THREADS);

        /**
         * Exclusive scan with Example 39-2: the Blelloch up-sweep / down-sweep
         * running in shared memory per tile, plus the same hierarchical offset
         * pass. This is the chapter's layout, whose stride-2 shared accesses cost
         * 2-way bank conflicts.
         */
        void scanEfficient(int n, int *odata, const int *idata,
                           int blockThreads = DEFAULT_BLOCK_THREADS);

        /**
         * Same tree scan with one pad slot per 32 shared elements, so the stride-2
         * tree accesses hit distinct banks. The benchmark uses it to quantify what
         * the bank conflicts of the chapter's layout actually cost.
         */
        void scanEfficientPadded(int n, int *odata, const int *idata,
                                 int blockThreads = DEFAULT_BLOCK_THREADS);
    }
}
