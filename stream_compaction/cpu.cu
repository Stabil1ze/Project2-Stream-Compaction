#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

		// CPU scan 
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int running = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = running;
                running += idata[i];
            }
            timer().endCpuTimer();
        }

        // CPU stream compaction without using the scan function
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        // CPU stream compaction using scan and scatter
        int compactWithScan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return 0;
            }

            int *bools = new int[n];
            int *indices = new int[n];

            timer().startCpuTimer();

            // Map
            for (int i = 0; i < n; ++i) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }

            // Scan
            int running = 0;
            for (int i = 0; i < n; ++i) {
                indices[i] = running;
                running += bools[i];
            }

            int count = indices[n - 1] + bools[n - 1];

            // Scatter
            for (int i = 0; i < n; ++i) {
                if (bools[i]) {
                    odata[indices[i]] = idata[i];
                }
            }

            timer().endCpuTimer();

            delete[] bools;
            delete[] indices;
            return count;
        }
    }
}
