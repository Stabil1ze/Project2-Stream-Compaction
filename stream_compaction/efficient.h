#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        /**
         * In-place exclusive scan of a device array of m elements, where m must
         * be a power of two and every element past the logical length must already
         * be zero. Exposed for GPU modules that build on the scan (extra credit 1:
         * radix sort).
         */
        void scanDevice(int m, int *data);
    }
}
