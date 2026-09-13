/**
 * @file      shared_scan.cu
 * @brief     Shared-memory scan (extra credit 2): GPU Gems 3, Chapter 39.
 *
 * The tile scans of Examples 39-1 (Hillis-Steele) and 39-2 (Blelloch) run with
 * one element per thread and the tile held in *dynamic* shared memory. Arrays
 * larger than one tile are handled the way the chapter describes: every block
 * scans its tile and records the tile total, the totals are scanned with one of
 * the project's scans, and a third kernel adds the resulting offsets back.
 *
 * The chapter's tree layout walks nodes with a stride of 2 * offset, which makes
 * two threads hit the same shared-memory bank; the padded variant inserts one
 * pad slot every 32 elements so those accesses land on distinct banks. The
 * benchmark measures both so the conflict cost is not just a claim.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include "shared_scan.h"

namespace StreamCompaction {
    namespace SharedScan {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer() {
            static PerformanceTimer timer;
            return timer;
        }

        namespace {

            const int MIN_BLOCK_THREADS = 32;
            const int MAX_BLOCK_THREADS = 1024;

            int nextPow2(int n) {
                return (n <= 1) ? 1 : (1 << ilog2ceil(n));
            }

            // The tile scans loop over the block until its size is covered, so
            // blockDim.x must be a power of two. Clamp and round up.
            int sanitizeBlockThreads(int blockThreads) {
                return std::min(std::max(nextPow2(blockThreads), MIN_BLOCK_THREADS),
                                MAX_BLOCK_THREADS);
            }

            // One pad slot per row of 32 elements (dynamic shared memory).
            int paddedElements(int b) {
                return b + ((b + 31) >> 5);
            }

            enum Variant {
                VARIANT_NAIVE = 0,        // Example 39-1
                VARIANT_TREE = 1,         // Example 39-2
                VARIANT_TREE_PADDED = 2   // Example 39-2, bank conflicts removed
            };
        }  // namespace

        /**
         * Example 39-1: Hillis-Steele scan of one tile in shared memory, one
         * element per thread. The intermediate values are read into registers
         * before the barrier, so a single shared array is enough, and every
         * access is stride 1 (no bank conflicts).
         *
         * Writes the exclusive scan of the tile back into data and the tile total
         * into tileSums[blockIdx.x].
         */
        __global__ void kernTileScanNaive(int n, int *data, int *tileSums) {
            extern __shared__ int temp[];
            const int t = threadIdx.x;
            const int b = blockDim.x;
            const int i = blockIdx.x * b + t;

            temp[t] = (i < n) ? data[i] : 0;
            __syncthreads();

            for (int d = 1; d < b; d <<= 1) {
                const int value = temp[t];
                const int previous = (t >= d) ? temp[t - d] : 0;
                __syncthreads();   // every thread has read its inputs
                temp[t] = value + previous;
                __syncthreads();   // ... and the new values are visible
            }

            if (t == b - 1) {
                tileSums[blockIdx.x] = temp[b - 1];
            }
            __syncthreads();

            if (i < n) {
                data[i] = (t == 0) ? 0 : temp[t - 1];
            }
        }

        /**
         * Example 39-2: Blelloch up-sweep / down-sweep of one tile in shared
         * memory. PADDED selects between the chapter's layout (stride-2 node
         * accesses, 2-way bank conflicts) and the padded one.
         */
        template<bool PADDED>
        __global__ void kernTileScanTree(int n, int *data, int *tileSums) {
            extern __shared__ int temp[];
            const int t = threadIdx.x;
            const int b = blockDim.x;
            const int i = blockIdx.x * b + t;
            const int pad = PADDED ? 1 : 0;

            temp[t + (t >> 5) * pad] = (i < n) ? data[i] : 0;
            __syncthreads();

            // Up-sweep: node (t + 1) * 2 * offset - 1 absorbs its left sibling.
            for (int offset = 1; offset < b; offset <<= 1) {
                if (t < b / (2 * offset)) {
                    const int node = (t + 1) * (2 * offset) - 1;
                    const int left = node - offset;
                    temp[node + (node >> 5) * pad] += temp[left + (left >> 5) * pad];
                }
                __syncthreads();
            }

            // The root is the tile total; zero it to turn the inclusive tree into
            // an exclusive scan before walking back down.
            if (t == 0) {
                tileSums[blockIdx.x] = temp[b - 1 + ((b - 1) >> 5) * pad];
                temp[b - 1 + ((b - 1) >> 5) * pad] = 0;
            }
            __syncthreads();

            // Down-sweep.
            for (int offset = b / 2; offset > 0; offset >>= 1) {
                if (t < b / (2 * offset)) {
                    const int node0 = (2 * t + 1) * offset - 1;
                    const int node1 = node0 + offset;
                    int &first = temp[node0 + (node0 >> 5) * pad];
                    int &second = temp[node1 + (node1 >> 5) * pad];
                    second += first;
                    first = second - first;
                }
                __syncthreads();
            }

            if (i < n) {
                data[i] = temp[t + (t >> 5) * pad];
            }
        }

        // Adds the exclusive prefix of the tile totals to every tile.
        __global__ void kernAddTileOffsets(int n, int *data, const int *tileOffsets) {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                data[i] += tileOffsets[blockIdx.x];
            }
        }

        namespace {

            size_t sharedBytes(Variant variant, int b) {
                const int elements = (variant == VARIANT_TREE_PADDED) ? paddedElements(b) : b;
                return static_cast<size_t>(elements) * sizeof(int);
            }

            void scanHierarchical(int n, int *odata, const int *idata, int blockThreads,
                                  Variant variant) {
                if (n <= 0) {
                    return;
                }

                const int b = sanitizeBlockThreads(blockThreads);
                const int numTiles = (n + b - 1) / b;
                const int tileSumCount = nextPow2(numTiles);

                int *devData = nullptr;
                int *devTileOffsets = nullptr;
                cudaMalloc(reinterpret_cast<void **>(&devData), n * sizeof(int));
                cudaMalloc(reinterpret_cast<void **>(&devTileOffsets), tileSumCount * sizeof(int));
                cudaMemcpy(devData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
                if (numTiles < tileSumCount) {
                    // The tile totals are scanned as a power-of-two array; the tile
                    // kernels only write the first numTiles entries.
                    cudaMemset(devTileOffsets + numTiles, 0,
                               (tileSumCount - numTiles) * sizeof(int));
                }
                checkCUDAError("SharedScan::scan: cudaMalloc / cudaMemcpy(H2D) / cudaMemset failed");

                timer().startGpuTimer();

                const size_t smem = sharedBytes(variant, b);
                switch (variant) {
                    case VARIANT_NAIVE:
                        kernTileScanNaive<<<numTiles, b, smem>>>(n, devData, devTileOffsets);
                        break;
                    case VARIANT_TREE:
                        kernTileScanTree<false><<<numTiles, b, smem>>>(n, devData, devTileOffsets);
                        break;
                    default:
                        kernTileScanTree<true><<<numTiles, b, smem>>>(n, devData, devTileOffsets);
                        break;
                }

                // Scan of the tile totals: the project's work-efficient scan, run
                // in place on device memory (the chapter's "scan the sums" step).
                Efficient::scanDevice(tileSumCount, devTileOffsets);

                kernAddTileOffsets<<<numTiles, b>>>(n, devData, devTileOffsets);

                timer().endGpuTimer();
                checkCUDAError("SharedScan::scan: kernel execution failed");

                cudaMemcpy(odata, devData, n * sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("SharedScan::scan: cudaMemcpy(D2H) failed");
                cudaFree(devData);
                cudaFree(devTileOffsets);
            }
        }  // namespace

        void scanNaive(int n, int *odata, const int *idata, int blockThreads) {
            scanHierarchical(n, odata, idata, blockThreads, VARIANT_NAIVE);
        }

        void scanEfficient(int n, int *odata, const int *idata, int blockThreads) {
            scanHierarchical(n, odata, idata, blockThreads, VARIANT_TREE);
        }

        void scanEfficientPadded(int n, int *odata, const int *idata, int blockThreads) {
            scanHierarchical(n, odata, idata, blockThreads, VARIANT_TREE_PADDED);
        }
    }
}
