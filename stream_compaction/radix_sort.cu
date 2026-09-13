/**
 * @file      radix_sort.cu
 * @brief     GPU radix sort for the stream compaction module (extra credit 1).
 *
 * One counting pass per 8-bit digit, least significant digit first:
 *   1. kernRadixHistogram builds a per-block digit histogram in global memory,
 *      in bin-major layout (hist[digit * numBlocks + block]).
 *   2. The project's work-efficient scan (Efficient::scanDevice) turns that
 *      histogram into the global start offset of every (digit, block) pair.
 *      In bin-major layout the exclusive prefix sum at (digit, block) is
 *      exactly "all elements with a smaller digit, plus the elements of this
 *      digit in the preceding blocks".
 *   3. kernRadixScatter computes each element's stable rank inside its block
 *      (warp-level shuffle ranking) and writes it to base + rank.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "radix_sort.h"

namespace StreamCompaction {
    namespace RadixSort {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

        namespace {

            const int RADIX_BITS = 8;                       // bits per counting pass
            const int RADIX = 1 << RADIX_BITS;              // 256 buckets
            const int PASSES = 32 / RADIX_BITS;             // 4 passes sort a whole int

            const int WARP_SIZE = 32;
            const int WARP_LOG2 = 5;
            const int BLOCK_THREADS = 256;
            const int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE;              // 8
            const int TILES_PER_WARP = 16;
            const int ELEMENTS_PER_WARP = TILES_PER_WARP * WARP_SIZE;           // 512
            const int ELEMENTS_PER_BLOCK = WARPS_PER_BLOCK * ELEMENTS_PER_WARP; // 4096

            // ilog2ceil comes from common.h.
            int nextPow2(int n) {
                return (n <= 1) ? 1 : (1 << ilog2ceil(n));
            }
        }  // namespace

        /**
         * Digit of `value` for the pass starting at bit `shift`. The sign bit is
         * flipped first so that the unsigned digit order equals the order of the
         * signed ints, which is what makes the sort work for negative values.
         */
        __device__ __forceinline__ int radixDigit(int value, int shift) {
            unsigned key = static_cast<unsigned>(value) ^ 0x80000000u;
            return static_cast<int>((key >> shift) & (RADIX - 1));
        }

        /**
         * Stable rank of `digit` within the calling warp: how many lanes with a
         * smaller lane index carry the same digit (rank), and how many lanes
         * carry it at all (count). A shuffle loop is used instead of
         * __match_any_sync so that the code also compiles for compute
         * capability < 7.0.
         */
        __device__ __forceinline__ void warpStableRank(int digit, int lane, int &rank, int &count) {
            rank = 0;
            count = 0;
            #pragma unroll
            for (int l = 0; l < WARP_SIZE; ++l) {
                const int other = __shfl_sync(0xffffffffu, digit, l);
                if (other == digit) {
                    ++count;
                    rank += (l < lane) ? 1 : 0;
                }
            }
        }

        /**
         * Counts the digit histogram of this warp's chunk of the block into
         * warpCount[warp * RADIX .. warp * RADIX + RADIX - 1] (shared memory).
         * Each warp owns a contiguous chunk, so the per-warp counts can be
         * turned into per-warp offsets without losing the input order.
         */
        __device__ __forceinline__ void countWarpChunk(int n, int shift, const int *idata, int *warpCount) {
            const int warp = threadIdx.x >> WARP_LOG2;
            const int lane = threadIdx.x & (WARP_SIZE - 1);
            int *counts = warpCount + warp * RADIX;

            for (int d = lane; d < RADIX; d += WARP_SIZE) {
                counts[d] = 0;
            }
            __syncthreads();

            const int chunkStart = blockIdx.x * ELEMENTS_PER_BLOCK + warp * ELEMENTS_PER_WARP;
            const int chunkEnd = min(chunkStart + ELEMENTS_PER_WARP, n);
            const int tiles = (chunkEnd - chunkStart + WARP_SIZE - 1) / WARP_SIZE;

            for (int t = 0; t < tiles; ++t) {
                const int i = chunkStart + t * WARP_SIZE + lane;
                if (i < chunkEnd) {
                    atomicAdd(&counts[radixDigit(idata[i], shift)], 1);
                }
            }
            __syncthreads();
        }

        /**
         * 256-bin digit histogram of every block, stored bin-major so that a
         * single exclusive scan of the array yields all global start offsets.
         */
        __global__ void kernRadixHistogram(int n, int shift, int numBlocks, const int *idata, int *hist) {
            __shared__ int warpCount[WARPS_PER_BLOCK * RADIX];
            countWarpChunk(n, shift, idata, warpCount);

            // One bin per thread: sum the warps of this block and store the
            // block total in hist[digit * numBlocks + block].
            for (int d = threadIdx.x; d < RADIX; d += BLOCK_THREADS) {
                int total = 0;
                for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                    total += warpCount[w * RADIX + d];
                }
                hist[d * numBlocks + blockIdx.x] = total;
            }
        }

        /**
         * Stable scatter: every element goes to
         * hist[digit * numBlocks + block] + (offsets of the preceding warps of
         * this block) + (stable rank inside its warp).
         */
        __global__ void kernRadixScatter(int n, int shift, int numBlocks, const int *idata,
                                         int *odata, const int *hist) {
            __shared__ int warpCount[WARPS_PER_BLOCK * RADIX];
            const int warp = threadIdx.x >> WARP_LOG2;
            const int lane = threadIdx.x & (WARP_SIZE - 1);

            countWarpChunk(n, shift, idata, warpCount);

            // Reuse the same counts as the exclusive offset of each warp, so
            // that a warp's elements follow the previous warps' elements.
            for (int d = threadIdx.x; d < RADIX; d += BLOCK_THREADS) {
                int running = 0;
                for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                    const int c = warpCount[w * RADIX + d];
                    warpCount[w * RADIX + d] = running;
                    running += c;
                }
            }
            __syncthreads();

            const int chunkStart = blockIdx.x * ELEMENTS_PER_BLOCK + warp * ELEMENTS_PER_WARP;
            const int chunkEnd = min(chunkStart + ELEMENTS_PER_WARP, n);
            const int tiles = (chunkEnd - chunkStart + WARP_SIZE - 1) / WARP_SIZE;

            for (int t = 0; t < tiles; ++t) {
                const int i = chunkStart + t * WARP_SIZE + lane;
                const bool active = i < chunkEnd;
                // Lanes without an element carry a digit that no bucket can have
                // (RADIX), so they never match a real lane in the ranking below
                // and the guarded store keeps every array index in range.
                const int digit = active ? radixDigit(idata[i], shift) : RADIX;

                int rank = 0;
                int count = 0;
                warpStableRank(digit, lane, rank, count);

                if (active) {
                    const int pos = hist[digit * numBlocks + blockIdx.x] +
                                    warpCount[warp * RADIX + digit] + rank;
                    odata[pos] = idata[i];
                    if (rank == 0) {  // first lane of this digit: advance the counter
                        atomicAdd(&warpCount[warp * RADIX + digit], count);
                    }
                }
                __syncwarp();
            }
        }

        void sort(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            const int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
            const int histSize = RADIX * numBlocks;
            const int histPadded = nextPow2(histSize);

            int *devA = nullptr;
            int *devB = nullptr;
            int *devHist = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&devA), n * sizeof(int));
            cudaMalloc(reinterpret_cast<void **>(&devB), n * sizeof(int));
            cudaMalloc(reinterpret_cast<void **>(&devHist), histPadded * sizeof(int));
            cudaMemcpy(devA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            if (histSize < histPadded) {
                // The scan treats the histogram as a power-of-two array. Only
                // the padding has to be zeroed: the histogram kernel rewrites
                // every entry below histSize on every pass.
                cudaMemset(devHist + histSize, 0, (histPadded - histSize) * sizeof(int));
            }
            checkCUDAError("RadixSort::sort: cudaMalloc / cudaMemcpy(H2D) / cudaMemset failed");

            timer().startGpuTimer();

            int *src = devA;
            int *dst = devB;
            for (int pass = 0; pass < PASSES; ++pass) {
                const int shift = pass * RADIX_BITS;

                kernRadixHistogram<<<numBlocks, BLOCK_THREADS>>>(n, shift, numBlocks, src, devHist);

                // Exclusive scan of the bin-major histogram: one of the scan
                // implementations of this project, run in place in device memory.
                StreamCompaction::Efficient::scanDevice(histPadded, devHist);

                kernRadixScatter<<<numBlocks, BLOCK_THREADS>>>(n, shift, numBlocks, src, dst, devHist);

                int *tmp = src;
                src = dst;
                dst = tmp;
            }

            timer().endGpuTimer();
            checkCUDAError("RadixSort::sort: kernel execution failed");

            cudaMemcpy(odata, src, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("RadixSort::sort: cudaMemcpy(D2H) failed");

            cudaFree(devA);
            cudaFree(devB);
            cudaFree(devHist);
        }
    }
}