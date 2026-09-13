// CIS 5650 Project 2 - Part 5 benchmark harness.
//
// Measures the scan implementations of this project, plus a "lazy" reference
// tree scan that launches the full m/2-thread grid on every level. That is the
// scheme described by the Part 5 hints: most threads are idle at the deeper
// levels and exit early through the index guard. The library version compacts
// the threads instead (only ceil(active / BLOCK) blocks are launched, and the
// thread index is mapped directly onto the merge index), so comparing the two
// quantifies the optimization.
//
// Usage:
//   cis5650_stream_compaction_bench.exe [--sizes 12,14,16] [--iters 30]
//                                       [--reps 3] [--csv] [--sort]
//
// All timings are given in milliseconds and cover GPU kernel work only (the
// same convention as the test program): device allocations and host/device
// copies are outside the timed region.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <stream_compaction/cpu.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/radix_sort.h>
#include <stream_compaction/shared_scan.h>
#include <stream_compaction/thrust.h>

namespace {

struct CudaTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void begin() {
        cudaEventRecord(start);
    }
    float endMs() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

float medianOf(std::vector<float> values) {
    if (values.empty()) {
        return 0.0f;
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

float minOf(const std::vector<float> &values) {
    if (values.empty()) {
        return 0.0f;
    }
    return *std::min_element(values.begin(), values.end());
}

int nextPow2(int n) {
    int m = 1;
    while (m < n) {
        m <<= 1;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Lazy level-by-level Blelloch scan (Part 5 baseline).
// Every level launches the same full grid of m/2 threads; the threads whose
// node does not exist at this level simply fall out through the guard.
// ---------------------------------------------------------------------------
const int LAZY_BLOCK_SIZE = 64;

__global__ void lazyUpSweep(int m, int offset, int *data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int active = m / (2 * offset);
    // The grid always covers m/2 threads; threads that have no node on this
    // level terminate early. The index is only computed for active threads so
    // that it cannot overflow at the top level.
    if (index < active) {
        int idx = (index + 1) * (2 * offset) - 1;
        data[idx] += data[idx - offset];
    }
}

__global__ void lazyDownSweep(int m, int offset, int *data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int active = m / (2 * offset);
    if (index < active) {
        int node0 = (2 * index + 1) * offset - 1;
        int node1 = node0 + offset;
        data[node1] += data[node0];
        data[node0] = data[node1] - data[node0];
    }
}

__global__ void lazySetLastZero(int m, int *data) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        data[m - 1] = 0;
    }
}

__global__ void emptyKernel() {}

void lazyTreeScanDevice(int m, int *data) {
    const int blocks = (m / 2 + LAZY_BLOCK_SIZE - 1) / LAZY_BLOCK_SIZE;
    for (int offset = 1; offset < m; offset <<= 1) {
        lazyUpSweep<<<blocks, LAZY_BLOCK_SIZE>>>(m, offset, data);
    }
    lazySetLastZero<<<1, 1>>>(m, data);
    for (int offset = m / 2; offset > 0; offset >>= 1) {
        lazyDownSweep<<<blocks, LAZY_BLOCK_SIZE>>>(m, offset, data);
    }
}

// ---------------------------------------------------------------------------
// Compacted per-level Blelloch scan (the library scheme before Part 5).
// Only ceil(active / BLOCK) blocks are launched per level and the thread index
// maps directly onto the merge index, but every level still costs a launch.
// ---------------------------------------------------------------------------
const int COMPACT_BLOCK_SIZE = 64;

__global__ void compactedUpSweep(int m, int offset, int *data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int active = m / (2 * offset);
    if (index < active) {
        int idx = (index + 1) * (2 * offset) - 1;
        data[idx] += data[idx - offset];
    }
}

__global__ void compactedDownSweep(int m, int offset, int *data) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int active = m / (2 * offset);
    if (index < active) {
        int node0 = (2 * index + 1) * offset - 1;
        int node1 = node0 + offset;
        data[node1] += data[node0];
        data[node0] = data[node1] - data[node0];
    }
}

__global__ void compactedSetLastZero(int m, int *data) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        data[m - 1] = 0;
    }
}

void perLevelTreeScanDevice(int m, int *data) {
    for (int offset = 1; offset < m; offset <<= 1) {
        int active = m / (2 * offset);
        int blocks = (active + COMPACT_BLOCK_SIZE - 1) / COMPACT_BLOCK_SIZE;
        compactedUpSweep<<<blocks, COMPACT_BLOCK_SIZE>>>(m, offset, data);
    }
    compactedSetLastZero<<<1, 1>>>(m, data);
    for (int offset = m / 2; offset > 0; offset >>= 1) {
        int active = m / (2 * offset);
        int blocks = (active + COMPACT_BLOCK_SIZE - 1) / COMPACT_BLOCK_SIZE;
        compactedDownSweep<<<blocks, COMPACT_BLOCK_SIZE>>>(m, offset, data);
    }
}

struct Options {
    std::vector<int> log2sizes;
    int iters;
    int reps;
    bool csv;
    bool sortMode;
    bool smemMode;
};

void fillRandom(std::vector<int> &data, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 99);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dist(rng);
    }
}

bool verify(const char *name, int n, const std::vector<int> &expected,
            const std::vector<int> &actual) {
    for (int i = 0; i < n; ++i) {
        if (expected[i] != actual[i]) {
            printf("VERIFY FAIL %s at n=%d index %d: expected %d got %d\n",
                   name, n, i, expected[i], actual[i]);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Timing helpers. Each returns iters * reps samples, which the caller reduces
// to a median. The library timers are used for the library implementations so
// that this harness measures exactly the same region as the test program.
// ---------------------------------------------------------------------------
std::vector<float> timeCpu(int n, const std::vector<int> &input,
                           std::vector<int> &scratch, int total) {
    std::vector<float> samples;
    samples.reserve(total);
    for (int k = 0; k < total; ++k) {
        StreamCompaction::CPU::scan(n, scratch.data(), input.data());
        samples.push_back(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation());
    }
    return samples;
}

std::vector<float> timeNaive(int n, const std::vector<int> &input,
                             std::vector<int> &scratch, int total) {
    std::vector<float> samples;
    samples.reserve(total);
    for (int k = 0; k < total; ++k) {
        StreamCompaction::Naive::scan(n, scratch.data(), input.data());
        samples.push_back(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation());
    }
    return samples;
}

std::vector<float> timeEfficient(int n, const std::vector<int> &input,
                                 std::vector<int> &scratch, int total) {
    std::vector<float> samples;
    samples.reserve(total);
    for (int k = 0; k < total; ++k) {
        StreamCompaction::Efficient::scan(n, scratch.data(), input.data());
        samples.push_back(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation());
    }
    return samples;
}

std::vector<float> timeThrust(int n, const std::vector<int> &input,
                              std::vector<int> &scratch, int total) {
    std::vector<float> samples;
    samples.reserve(total);
    for (int k = 0; k < total; ++k) {
        StreamCompaction::Thrust::scan(n, scratch.data(), input.data());
        samples.push_back(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation());
    }
    return samples;
}

std::vector<float> timePerLevel(int n, const std::vector<int> &input, int total) {
    const int m = nextPow2(n);
    std::vector<int> padded(m, 0);
    std::copy(input.begin(), input.end(), padded.begin());

    int *dev = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&dev), m * sizeof(int));
    checkCUDAError("bench: cudaMalloc failed");

    CudaTimer timer;
    std::vector<float> samples;
    samples.reserve(total);
    for (int k = 0; k < total; ++k) {
        cudaMemcpy(dev, padded.data(), m * sizeof(int), cudaMemcpyHostToDevice);
        timer.begin();
        perLevelTreeScanDevice(m, dev);
        samples.push_back(timer.endMs());
    }
    cudaFree(dev);
    return samples;
}
std::vector<float> timeLazy(int n, const std::vector<int> &input, int total) {
    const int m = nextPow2(n);
    std::vector<int> padded(m, 0);
    std::copy(input.begin(), input.end(), padded.begin());

    int *dev = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&dev), m * sizeof(int));
    checkCUDAError("bench: cudaMalloc failed");

    CudaTimer timer;
    std::vector<float> samples;
    samples.reserve(total);
    for (int k = 0; k < total; ++k) {
        cudaMemcpy(dev, padded.data(), m * sizeof(int), cudaMemcpyHostToDevice);
        timer.begin();
        lazyTreeScanDevice(m, dev);
        samples.push_back(timer.endMs());
    }
    cudaFree(dev);
    return samples;
}


// ---------------------------------------------------------------------------
// Extra credit 1: radix sort benchmark (std::sort / RadixSort / thrust::sort).
// std::sort is sampled fewer times because it takes hundreds of milliseconds
// per sort on the largest arrays.
// ---------------------------------------------------------------------------
const int SORT_CPU_SAMPLES = 5;
const int SORT_GPU_SAMPLES = 20;

void runSortBenchmark(const Options &opt, const cudaDeviceProp &prop) {
    printf("CIS 5650 Project 2 - radix sort benchmark (extra credit 1)\n");
    printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("samples per point: %d (std::sort), %d (GPU sorts)\n\n", SORT_CPU_SAMPLES, SORT_GPU_SAMPLES);
    printf("%-11s %10s %10s %10s   %s\n", "n", "std::sort", "RadixSort", "thrust", "check");

    for (int lg : opt.log2sizes) {
        const int n = 1 << lg;
        std::vector<int> input(n), reference(n), scratch(n);
        fillRandom(input, 0x9E3779B9u ^ static_cast<unsigned int>(lg));
        for (int i = 0; i < n; i += 3) {
            input[i] = -input[i] - 1;  // exercise the signed path
        }
        reference = input;
        std::sort(reference.begin(), reference.end());

        std::fill(scratch.begin(), scratch.end(), 0);
        StreamCompaction::RadixSort::sort(n, scratch.data(), input.data());
        const bool okRadix = verify("RadixSort", n, reference, scratch);

        StreamCompaction::RadixSort::sort(n, scratch.data(), input.data());  // warm up
        (void)std::sort(scratch.begin(), scratch.end());

        std::vector<float> cpuSamples;
        std::vector<float> radixSamples;
        std::vector<float> thrustSamples;
        for (int k = 0; k < SORT_CPU_SAMPLES; ++k) {
            scratch = input;
            const std::chrono::high_resolution_clock::time_point t0 = std::chrono::high_resolution_clock::now();
            std::sort(scratch.begin(), scratch.end());
            const std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
            cpuSamples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        for (int k = 0; k < SORT_GPU_SAMPLES; ++k) {
            StreamCompaction::RadixSort::sort(n, scratch.data(), input.data());
            radixSamples.push_back(StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation());
        }

        thrust::device_vector<int> d_input(input.begin(), input.end());
        thrust::device_vector<int> d_work(n);
        CudaTimer thrustTimer;
        for (int k = 0; k < SORT_GPU_SAMPLES; ++k) {
            thrust::copy(d_input.begin(), d_input.end(), d_work.begin());  // outside the timed region
            thrustTimer.begin();
            thrust::sort(d_work.begin(), d_work.end());
            thrustSamples.push_back(thrustTimer.endMs());
        }
        thrust::copy(d_work.begin(), d_work.end(), scratch.begin());
        const bool okThrust = verify("ThrustSort", n, reference, scratch);

        printf("%-11d %10.4f %10.4f %10.4f    %s\n", n,
               medianOf(cpuSamples), medianOf(radixSamples), medianOf(thrustSamples),
               (okRadix && okThrust) ? "ok" : "VERIFY FAIL");
    }
}

// ---------------------------------------------------------------------------
// Extra credit 2: shared-memory scan (GPU Gems 39). Variant 0 = Example 39-1,
// 1 = Example 39-2 with the chapter's shared layout, 2 = Example 39-2 padded.
// ---------------------------------------------------------------------------
std::vector<float> timeSharedScan(int n, const std::vector<int> &input,
                                  std::vector<int> &scratch, int total, int variant,
                                  int blockThreads) {
    std::vector<float> samples;
    samples.reserve(total);
    for (int k = 0; k < total; ++k) {
        switch (variant) {
            case 0:
                StreamCompaction::SharedScan::scanNaive(n, scratch.data(), input.data(), blockThreads);
                break;
            case 1:
                StreamCompaction::SharedScan::scanEfficient(n, scratch.data(), input.data(), blockThreads);
                break;
            default:
                StreamCompaction::SharedScan::scanEfficientPadded(n, scratch.data(), input.data(), blockThreads);
                break;
        }
        samples.push_back(
            StreamCompaction::SharedScan::timer().getGpuElapsedTimeForPreviousOperation());
    }
    return samples;
}

// Repeats the shared-memory access pattern of the tree scan, so that the cost of
// the bank conflicts is visible even though the real scan is dominated by other
// work. PADDED selects between the chapter's layout and the padded one.
template<bool PADDED>
__global__ void kernBankConflictProbe(int repetitions, int *out) {
    extern __shared__ int temp[];
    const int t = threadIdx.x;
    const int b = blockDim.x;
    const int pad = PADDED ? 1 : 0;

    for (int i = t; i < b; i += b) {
        temp[i + (i >> 5) * pad] = i + 1;
    }
    __syncthreads();

    for (int r = 0; r < repetitions; ++r) {
        for (int offset = 1; offset < b; offset <<= 1) {
            if (t < b / (2 * offset)) {
                const int node = (t + 1) * (2 * offset) - 1;
                const int left = node - offset;
                temp[node + (node >> 5) * pad] += temp[left + (left >> 5) * pad];
            }
            __syncthreads();
        }
    }

    if (t == 0) {
        out[blockIdx.x] = temp[0];
    }
}

// Times the two shared-memory layouts of the tree pattern in isolation.
void runBankConflictProbe() {
    const int probes = 4096;
    const int block = 256;
    const int repetitions = 200;
    const int paddedElements = block + ((block + 31) >> 5);
    const size_t smemPadded = static_cast<size_t>(paddedElements) * sizeof(int);
    const size_t smemPlain = static_cast<size_t>(block) * sizeof(int);

    int *devOut = nullptr;
    cudaMalloc(reinterpret_cast<void **>(&devOut), probes * sizeof(int));
    checkCUDAError("bench: bank conflict probe cudaMalloc failed");

    CudaTimer timer;
    float times[2] = { 0.0f, 0.0f };
    // 0 = chapter layout, 1 = padded layout
    kernBankConflictProbe<false><<<probes, block, smemPlain>>>(repetitions, devOut);
    timer.begin();
    kernBankConflictProbe<false><<<probes, block, smemPlain>>>(repetitions, devOut);
    times[0] = timer.endMs();
    kernBankConflictProbe<true><<<probes, block, smemPadded>>>(repetitions, devOut);
    timer.begin();
    kernBankConflictProbe<true><<<probes, block, smemPadded>>>(repetitions, devOut);
    times[1] = timer.endMs();
    checkCUDAError("bench: bank conflict probe failed");

    cudaFree(devOut);

    printf("\nshared-memory access pattern in isolation (%d blocks, block %d, %d repeats)\n",
           probes, block, repetitions);
    printf("%-34s %12s %10s\n", "layout", "time (ms)", "relative");
    printf("%-34s %12.4f %10s\n", "Example 39-2 (stride-2, conflicts)", times[0], "1.00x");
    printf("%-34s %12.4f %10.2fx\n", "Example 39-2, one pad per 32 elements",
           times[1], times[1] / times[0]);
}

void runSmemBenchmark(const Options &opt, const cudaDeviceProp &prop) {
    printf("CIS 5650 Project 2 - shared-memory scan benchmark (extra credit 2)\n");
    printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("samples per point: %d iters x %d repetitions, median, block size %d\n\n",
           opt.iters, opt.reps, StreamCompaction::SharedScan::DEFAULT_BLOCK_THREADS);
    printf("%-11s %10s %10s %10s %10s %10s %10s %10s\n",
           "n", "CPU", "Eff(fused)", "Smem39-1", "Smem39-2", "39-2 pad", "Thrust", "check");

    const int total = opt.iters * opt.reps;
    bool ok = true;

    for (int lg : opt.log2sizes) {
        const int n = 1 << lg;
        std::vector<int> input(n), reference(n), scratch(n, 0);
        fillRandom(input, 0x9E3779B9u ^ static_cast<unsigned int>(lg));

        StreamCompaction::CPU::scan(n, reference.data(), input.data());

        // Correctness of every variant (this also warms the kernels up).
        for (int variant = 0; variant < 3; ++variant) {
            std::fill(scratch.begin(), scratch.end(), 0);
            (void)timeSharedScan(n, input, scratch, 1, variant,
                                 StreamCompaction::SharedScan::DEFAULT_BLOCK_THREADS);
            const char *name = (variant == 0) ? "SharedScan39-1"
                             : (variant == 1) ? "SharedScan39-2" : "SharedScan39-2padded";
            ok = verify(name, n, reference, scratch) && ok;
        }

        StreamCompaction::Efficient::scan(n, scratch.data(), input.data());
        ok = verify("Efficient", n, reference, scratch) && ok;

        const int block = StreamCompaction::SharedScan::DEFAULT_BLOCK_THREADS;
        const std::vector<float> cpu = timeCpu(n, input, scratch, total);
        const std::vector<float> eff = timeEfficient(n, input, scratch, total);
        const std::vector<float> naive = timeSharedScan(n, input, scratch, total, 0, block);
        const std::vector<float> tree = timeSharedScan(n, input, scratch, total, 1, block);
        const std::vector<float> padded = timeSharedScan(n, input, scratch, total, 2, block);
        const std::vector<float> thrustSamples = timeThrust(n, input, scratch, total);

        printf("%-11d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f    %s\n", n,
               medianOf(cpu), medianOf(eff), medianOf(naive), medianOf(tree),
               medianOf(padded), medianOf(thrustSamples), ok ? "ok" : "VERIFY FAIL");
    }

    // The tile lives in dynamic shared memory, so the block size decides both the
    // tile size and the shared memory per block (and therefore the number of
    // resident blocks per SM).
    const int sweepLg = opt.log2sizes.back();
    const int sweepN = 1 << sweepLg;
    std::vector<int> input(sweepN), scratch(sweepN, 0);
    fillRandom(input, 0x51ED2701u);
    printf("\nblock-size sweep at n = %d (%d samples each)\n",
           sweepN, opt.iters);
    printf("%-8s %14s %14s %14s %12s\n", "block", "39-1 (ms)", "39-2 pad (ms)",
           "shared bytes", "tiles");
    const int sweepBlocks[] = { 128, 256, 512, 1024 };
    for (int b : sweepBlocks) {
        (void)timeSharedScan(sweepN, input, scratch, 1, 0, b);
        (void)timeSharedScan(sweepN, input, scratch, 1, 2, b);
        const std::vector<float> naive = timeSharedScan(sweepN, input, scratch, opt.iters, 0, b);
        const std::vector<float> padded = timeSharedScan(sweepN, input, scratch, opt.iters, 2, b);
        const int paddedElements = b + ((b + 31) >> 5);
        printf("%-8d %14.4f %14.4f %14d %12d\n", b, medianOf(naive), medianOf(padded),
               paddedElements * static_cast<int>(sizeof(int)), (sweepN + b - 1) / b);
    }

    runBankConflictProbe();
}
}  // namespace

int main(int argc, char **argv) {
    Options opt;
    opt.log2sizes = {12, 14, 16, 18, 20, 22};
    opt.iters = 30;
    opt.reps = 3;
    opt.csv = false;
    opt.sortMode = false;
    opt.smemMode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv") {
            opt.csv = true;
        } else if (arg == "--sort") {
            opt.sortMode = true;
        } else if (arg == "--smem") {
            opt.smemMode = true;
        } else if (arg == "--iters" && i + 1 < argc) {
            opt.iters = std::atoi(argv[++i]);
        } else if (arg == "--reps" && i + 1 < argc) {
            opt.reps = std::atoi(argv[++i]);
        } else if (arg == "--sizes" && i + 1 < argc) {
            opt.log2sizes.clear();
            char *token = std::strtok(argv[++i], ",");
            while (token != nullptr) {
                opt.log2sizes.push_back(std::atoi(token));
                token = std::strtok(nullptr, ",");
            }
        } else {
            printf("unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    checkCUDAError("bench: cudaGetDeviceProperties failed");

    if (opt.sortMode) {
        runSortBenchmark(opt, prop);
        return 0;
    }

    if (opt.smemMode) {
        runSmemBenchmark(opt, prop);
        return 0;
    }

    if (!opt.csv) {
        printf("CIS 5650 Project 2 - Part 5 scan benchmark\n");
        printf("GPU: %s (compute capability %d.%d)\n", prop.name, prop.major, prop.minor);
        printf("samples per point: %d iters x %d repetitions\n\n", opt.iters, opt.reps);
    }

    const int launchCount = 200;
    CudaTimer launchTimer;
    std::vector<float> launchSamples;
    for (int r = 0; r < 5; ++r) {
        launchTimer.begin();
        for (int i = 0; i < launchCount; ++i) {
            emptyKernel<<<1, 32>>>();
        }
        launchSamples.push_back(launchTimer.endMs() * 1000.0f / launchCount);
    }
    const float launchUs = medianOf(launchSamples);

    if (opt.csv) {
        printf("# launch_us_per_kernel,%.4f\n", launchUs);
        printf("impl,log2n,n,median_ms,min_ms\n");
    } else {
        printf("empty kernel launch cost: %.2f us per launch (back-to-back)\n\n", launchUs);
        printf("%-11s %10s %10s %10s %10s %10s %10s\n",
               "n", "CPU", "Naive", "Lazy", "PerLevel", "Efficient", "Thrust");
    }

    const int total = opt.iters * opt.reps;
    bool ok = true;

    for (int lg : opt.log2sizes) {
        const int n = 1 << lg;
        const int m = nextPow2(n);
        const int launches = 2 * (lg + (m > n ? 1 : 0)) + 1;
        (void)launches;

        std::vector<int> input(n);
        std::vector<int> reference(n, 0);
        std::vector<int> scratch(n, 0);
        fillRandom(input, 0x9E3779B9u ^ static_cast<unsigned int>(lg));

        StreamCompaction::CPU::scan(n, reference.data(), input.data());

        // ---- correctness of every variant ----
        std::fill(scratch.begin(), scratch.end(), 0);
        StreamCompaction::Naive::scan(n, scratch.data(), input.data());
        ok = verify("Naive", n, reference, scratch) && ok;

        std::fill(scratch.begin(), scratch.end(), 0);
        StreamCompaction::Efficient::scan(n, scratch.data(), input.data());
        ok = verify("Efficient", n, reference, scratch) && ok;

        std::fill(scratch.begin(), scratch.end(), 0);
        StreamCompaction::Thrust::scan(n, scratch.data(), input.data());
        ok = verify("Thrust", n, reference, scratch) && ok;

        {
            int *dev = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&dev), m * sizeof(int));
            std::vector<int> padded(m, 0);
            std::copy(input.begin(), input.end(), padded.begin());
            cudaMemcpy(dev, padded.data(), m * sizeof(int), cudaMemcpyHostToDevice);
            perLevelTreeScanDevice(m, dev);
            std::fill(scratch.begin(), scratch.end(), 0);
            cudaMemcpy(scratch.data(), dev, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev);
            checkCUDAError("bench: per-level verification failed");
            ok = verify("PerLevel", n, reference, scratch) && ok;
        }

        {
            int *dev = nullptr;
            cudaMalloc(reinterpret_cast<void **>(&dev), m * sizeof(int));
            std::vector<int> padded(m, 0);
            std::copy(input.begin(), input.end(), padded.begin());
            cudaMemcpy(dev, padded.data(), m * sizeof(int), cudaMemcpyHostToDevice);
            lazyTreeScanDevice(m, dev);
            std::fill(scratch.begin(), scratch.end(), 0);
            cudaMemcpy(scratch.data(), dev, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev);
            checkCUDAError("bench: lazy verification failed");
            ok = verify("Lazy", n, reference, scratch) && ok;
        }

        // ---- warm up ----
        StreamCompaction::Naive::scan(n, scratch.data(), input.data());
        StreamCompaction::Efficient::scan(n, scratch.data(), input.data());
        StreamCompaction::Thrust::scan(n, scratch.data(), input.data());
        (void)timeLazy(n, input, 1);
        (void)timePerLevel(n, input, 1);

        // ---- measurements ----
        const std::vector<float> cpu = timeCpu(n, input, scratch, total);
        const std::vector<float> naive = timeNaive(n, input, scratch, total);
        const std::vector<float> eff = timeEfficient(n, input, scratch, total);
        const std::vector<float> perLevel = timePerLevel(n, input, total);
        const std::vector<float> lazy = timeLazy(n, input, total);
        const std::vector<float> thrust = timeThrust(n, input, scratch, total);

        const float cpuMs = medianOf(cpu);
        const float naiveMs = medianOf(naive);
        const float effMs = medianOf(eff);
        const float perLevelMs = medianOf(perLevel);
        const float lazyMs = medianOf(lazy);
        const float thrustMs = medianOf(thrust);

        if (opt.csv) {
            printf("CPU,%d,%d,%.6f,%.6f\n", lg, n, cpuMs, minOf(cpu));
            printf("Naive,%d,%d,%.6f,%.6f\n", lg, n, naiveMs, minOf(naive));
            printf("Efficient,%d,%d,%.6f,%.6f\n", lg, n, effMs, minOf(eff));
            printf("Lazy,%d,%d,%.6f,%.6f\n", lg, n, lazyMs, minOf(lazy));
            printf("PerLevel,%d,%d,%.6f,%.6f\n", lg, n, perLevelMs, minOf(perLevel));
            printf("Thrust,%d,%d,%.6f,%.6f\n", lg, n, thrustMs, minOf(thrust));
        } else {
            printf("%-11d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n",
                   n, cpuMs, naiveMs, lazyMs, perLevelMs, effMs, thrustMs);
        }
    }

    if (!ok) {
        printf("one or more verification checks failed\n");
        return 1;
    }
    return 0;
}
