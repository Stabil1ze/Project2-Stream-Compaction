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
}  // namespace

int main(int argc, char **argv) {
    Options opt;
    opt.log2sizes = {12, 14, 16, 18, 20, 22};
    opt.iters = 30;
    opt.reps = 3;
    opt.csv = false;
    opt.sortMode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv") {
            opt.csv = true;
        } else if (arg == "--sort") {
            opt.sortMode = true;
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