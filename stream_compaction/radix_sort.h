#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace RadixSort {

        StreamCompaction::Common::PerformanceTimer& timer();

        /**
         * Sorts `n` ints of idata (ascending) into odata with a stable LSD radix
         * sort built on the work-efficient scan.
         *
         * Device allocations and the host/device copies are done outside the
         * timed region, exactly like the other GPU implementations of this
         * project, so getGpuElapsedTimeForPreviousOperation() reports the four
         * counting passes (histogram + scan + scatter) only.
         *
         * Handles negative values (the sign bit is flipped, so the unsigned
         * digit order matches the order of the signed ints) and any n >= 1.
         */
        void sort(int n, int *odata, const int *idata);
    }
}