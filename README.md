# CUDA Stream Compaction

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 2 - Stream Compaction**

* **Jing Huang**
  * [GitHub](https://github.com/Stabil1ze)
* Tested on: Windows 11, Intel i7-12700H @ 2.30GHz 23GB,
  NVIDIA GeForce RTX 3060 Laptop GPU 6GB (Personal computer)

## Overview

**Scan** (exclusive prefix sum) and **stream compaction** (remove every `0` from
an `int` array) - the two building blocks the path tracer uses to drop terminated
rays. `src/bench.cu` and the code review were done with the help of AI agents.

| # | Feature | File |
|---|---|---|
| 1 | CPU scan, `compactWithoutScan`, `compactWithScan` (reference for every GPU test) | `cpu.cu` |
| 2 | Naive scan, GPU Gems 39.2.1: two global buffers, one kernel per doubling | `naive.cu` |
| 3 | Work-efficient scan + compaction (Blelloch, active tree nodes only) | `efficient.cu`, `common.cu` |
| 4 | Thrust scan wrapper (`thrust::exclusive_scan`) | `thrust.cu` |
| 5 | `checkCUDAError` after every CUDA call (11 sites, all outside the timed regions) | `naive/efficient/thrust.cu` |
| 6 | [Radix sort](#extra-credit-1-radix-sort) built on the scan - extra credit 1 | `radix_sort.{h,cu}` |
| 7 | [Shared-memory scan](#extra-credit-2-shared-memory-scan), GPU Gems 39 - extra credit 2 | `shared_scan.{h,cu}` |

Part 5 (extra credit, see [below](#part-5-extra-credit-5-why-is-my-work-efficient-gpu-scan-slower-than-the-cpu))
fuses the upper tree levels into one 1024-thread block, cutting the launches per
scan from `2*log2(m)+1` to `2*log2(m/2048)+3` (45 -> 25 at `n = 2^22`, 25 -> 5 at
`n = 2^12`). All GPU scans support **non-power-of-two** sizes by padding to the
next power of two and only reporting the first `n` results.

## Performance Analysis

`src/bench.cu` measures through the project's `PerformanceTimer`, so the timed
region matches the supplied tests: kernel work only, `cudaMalloc`/`cudaMemcpy`
excluded. Each cell is the median of three processes x 150 samples (50 scans x 3
repetitions) of random values in `[0, 100)`, and every variant is checked against
the CPU reference (see [CMakeLists.txt changes](#cmakeliststxt-changes)).
Release build, CUDA 13.3, RTX 3060 Laptop.

| n        | CPU scan (ms) | Naive scan (ms) | Work-efficient (ms) | Thrust (ms) |
|----------|--------------:|----------------:|--------------------:|------------:|
| 4,096    | 0.0015        | 0.0707          | 0.0236              | 0.0306      |
| 16,384   | 0.0057        | 0.0768          | 0.0379              | 0.0310      |
| 65,536   | 0.0226        | 0.1192          | 0.1163              | 0.0327      |
| 262,144  | 0.0884        | 0.1398          | 0.1420              | 0.3245      |
| 1,048,576| 0.3634        | 0.6803          | 0.3793              | 0.3459      |
| 4,194,304| 1.6967        | 2.8851          | 1.2528              | 0.5180      |

![Scan performance comparison](img/performance-scan.png)

### Block-size selection

Swept at `n = 2^22` (median of many runs):

| Implementation | Sizes tried | Best | Notes |
|---|---|---|---|
| Naive scan | 128-1024 | 512 | Memory-bound full-grid kernels; 512 balances blocks/SM against coalescing |
| Work-efficient, per-level | 64-256 | 64 | Deep levels launch few threads, so small blocks waste less |
| Work-efficient, fused block | 256-1024 | 1024 | Only the fused block changes the launch count |

Only the fused block changes the *number of kernel launches*, which is what
dominates at small `n`:

| `FUSED_BLOCK_SIZE` | launches @ `2^12` | time @ `2^12` (ms) | launches @ `2^22` | time @ `2^22` (ms) |
|---|---|---|---|---|
| 256 | 9 | 0.0508 | 29 | 1.341 |
| 512 | 7 | 0.0328 | 27 | 1.306 |
| 1024 | 5 | 0.0272 | 25 | 1.329 |

At `2^22` the three agree within noise; at `2^12` every extra launch costs ~5 us,
exactly the ~0.02 ms gap between the 5-launch and 9-launch runs.
(`50 x 3` medians; 512 once, the others twice.)

### Observations

* **CPU wins up to about `2^20`** (a tie there: 0.3634 vs 0.3793 ms); at `2^22`
  the work-efficient scan is 1.35x and Thrust 3.3x faster. The CPU is one linear
  pass over cache-friendly memory; a GPU scan pays tens of microseconds of launch
  overhead first.
* **Naive scales worst**: O(n log n), 22 full-array sweeps (~740 MB at `2^22`),
  and the slowest implementation at 4M (2.8851 ms).
* **Work-efficient moves only the active tree nodes** (~100 MB at `2^22`) but pays
  one launch per level; after Part 5 it is the fastest hand-written scan at 4M
  (1.2528 ms).
* **Thrust wins from `2^18`** (0.5180 ms at 4M, one kernel with a decoupled
  look-back); below ~64K its dispatch cost dominates (~0.03 ms) and it dips at
  `2^18`-`2^20` (0.32-0.35 ms).
* **Block size is a 10-30% effect** for the naive and per-level scans.

## Part 5 (extra credit, +5): why is my work-efficient GPU scan slower than the CPU?

Before Part 5 the work-efficient scan needed **0.1423 ms at `n = 2^12`** while
the serial CPU scan needed **0.0015 ms** - about 95x slower - and it stayed
behind the CPU until roughly `2^21` elements.

The harness separates the two suspects: `Lazy` launches the full `m/2`-thread
grid at every level and lets the guard drop the idle threads, `PerLevel` launches
only `ceil(active/64)` blocks per level, and `Fused` is this project's version.

| n | Lazy (ms) | Per-level (ms) | Fused (ms) |
|---|---|---|---|
| 4,096 | 0.1454 | 0.1423 | 0.0236 |
| 16,384 | 0.1587 | 0.1577 | 0.0379 |
| 65,536 | 0.1994 | 0.1889 | 0.1163 |
| 262,144 | 0.2686 | 0.2372 | 0.1420 |
| 1,048,576 | 0.6877 | 0.4651 | 0.3793 |
| 4,194,304 | 2.5238 | 1.3532 | 1.2528 |

Two measurements explain it. A level-by-level Blelloch scan issues `2*log2(m)+1`
kernels (25 at `2^12`, 45 at `2^22`) and a back-to-back empty launch costs
**6.9 us** here (5.4 - 8.9 us), so 25 launches account for essentially all of the
0.1423 ms measured; the tree work itself is a few microseconds. And compacting
the threads alone buys nothing: `Lazy` and `PerLevel` agree within 2%, because
launching one block costs the same as launching 32,768 - it only matters at
`2^22`, where the lazy grid has to *dispatch* 32,768 idle blocks per level
(1.9x slower, 2.5238 vs 1.3532 ms).

Per-launch accounting for the fused version:

| n | launches before | launches after | removed | time before (ms) | time after (ms) | speedup | implied cost per removed launch |
|---|---|---|---|---|---|---|---|
| 4,096 | 25 | 5 | 20 | 0.1423 | 0.0236 | **6.0x** | 5.9 us |
| 16,384 | 29 | 9 | 20 | 0.1577 | 0.0379 | **4.2x** | 6.0 us |
| 65,536 | 33 | 13 | 20 | 0.1889 | 0.1163 | 1.6x | 3.6 us |
| 262,144 | 37 | 17 | 20 | 0.2372 | 0.1420 | 1.7x | 4.8 us |
| 1,048,576 | 41 | 21 | 20 | 0.4651 | 0.3793 | 1.2x | 4.3 us |
| 4,194,304 | 45 | 25 | 20 | 1.3532 | 1.2528 | 1.08x | 5.0 us |

The last column is `(time before - time after) / 20` and lands near the measured
6.9 us at every size: the gain is exactly *removed launches x launch latency*.

### The optimization: fuse the upper tree levels into one block

Once a level fits in one block it needs no launch of its own - the same block
loops over the remaining levels, with `__syncthreads()` between them to respect
the tree's read-after-write dependencies.

```cpp
const int FUSED_BLOCK_SIZE = 1024;

// Up-sweep of every level whose merge count fits in one block
__global__ void kernEfficientUpSweepFused(int m, int firstOffset, int *data) {
    for (int offset = firstOffset; offset < m; offset <<= 1) {
        int active = m / (2 * offset);
        if (threadIdx.x < active) {
            int idx = (threadIdx.x + 1) * (2 * offset) - 1;
            data[idx] += data[idx - offset];
        }
        __syncthreads();
    }
}

// Down-sweep of the levels above lastOffset, same shape: it adds the left node
// to the right one and keeps the old left value, turning the inclusive tree into
// an exclusive scan (kernEfficientDownSweepFused in efficient.cu).
```

`scanDevice` splits the tree at `fusedFirst = m / (2 * FUSED_BLOCK_SIZE)`: levels
below stay per-level launches with the tuned block size, the levels above (which
used to run with 1 - 1024 active threads) run in one block. So `m <= 2048` needs
three launches and `2^22` needs 25: 11 bulk up-sweep, 1 fused up-sweep, the
zeroing, 1 fused down-sweep, 11 bulk down-sweep.

![Part 5: fusion and launch counts](img/performance-part5.png)

### Limits and correctness

The launches that disappear are the *cheap* ones: no memory traffic is removed,
so the ~5 us saved at `2^22` against a ~1.25 ms total is all this structure
allows; going further means removing launches (extra credit 2 below, or CUB's
single-pass look-back at 0.5180 ms). Integer overflow was a real trap -
`(index + 1) * (2 * offset)` reaches `2^31` for threads without a node, and
guarding only the store was not enough (compute-sanitizer caught the predicated
reads), so every kernel derives `active` first and computes the index inside the
guard. `src/bench.cu` re-checks all variants against the CPU reference, the
supplied tests pass 12/12 (`SIZE = 256` and `1 << 20`), compute-sanitizer reports
0 errors, and a forced failure (`CUDA_VISIBLE_DEVICES=999`) makes the new
`checkCUDAError` print the failing call and exit with status 1.

## Extra Credit 1: Radix Sort

`radix_sort.{h,cu}`: a **stable 8-bit LSD radix sort** built on the work-efficient
scan, so four counting passes sort a whole 32-bit `int`:

```cpp
#include <stream_compaction/radix_sort.h>

int input[8]  = { 5, -3, 0, 128, 5, 7, -1000, 2 };
int output[8];
StreamCompaction::RadixSort::sort(8, output, input);
// input  [   5  -3   0 128   5   7 -1000   2 ]
// output [ -1000  -3   0   2   5   5   7 128 ]
```

One pass per digit, least significant byte first:

1. `kernRadixHistogram` - per-block 256-bin histogram in shared memory (one warp per 512-element chunk), written bin-major as `hist[digit * numBlocks + block]`.
2. `Efficient::scanDevice` - the exclusive scan of that array *is* the start offset of every (digit, block) pair.
3. `kernRadixScatter` - each element adds its per-warp offset and stable warp-local rank to that base.

Blocks, warps and lanes stay in input order, so every pass is stable and the four
passes sort completely; negatives work by flipping the sign bit. The histogram is
padded to the next power of two.

Mixed-sign random input (every third element negated); the GPU time covers the
four passes only, `std::sort` is sampled 5 times and the GPU 20 times per size.

| n | std::sort (ms) | RadixSort (ms) | thrust::sort (ms) | speedup vs std::sort |
|---|---|---|---|---|
| 4,096 | 0.1129 | 0.1772 | 0.0594 | 0.64x |
| 16,384 | 0.4604 | 0.2222 | 0.1640 | 2.1x |
| 65,536 | 1.6824 | 0.3132 | 0.1208 | 5.4x |
| 262,144 | 6.8671 | 0.4188 | 0.4864 | 16.4x |
| 1,048,576 | 28.0361 | 0.9173 | 0.7680 | 30.6x |
| 4,194,304 | 113.5395 | 2.3994 | 1.8555 | 47.3x |

Single-process medians (10-30% spread below 262K). The kernels use 8 KB of shared
memory and 26/40 registers with no spills (6 blocks / 1536 threads per SM). At
`2^12` the sort issues 20 launches, ~0.12 ms of the 0.177 ms measured - the Part 5
story again; from ~16K elements the O(n) counting passes win, reaching 47x over
`std::sort` at 4M, while `thrust::sort` stays 1.3x ahead (one kernel per pass
instead of histogram / scan / scatter). Reproduce with
`stream_compaction_bench.exe --sort`.

**Tests:** 13 cases compared element by element against `std::sort` (the sorted
sequence is unique, so duplicates are covered): 256/253, 1024 signed values,
all-equal, sorted, reversed, `INT_MIN`/`INT_MAX` extremes, a single element, 20000
elements with 3 distinct keys across 5 blocks, 12345 multi-block
non-power-of-two, and `2^20` covering the whole 32-bit range. All pass,
compute-sanitizer reports 0 errors, and a temporary fuzz harness matched
`std::sort` on 335/335 cases (`n` in `[1, 60000]` over six distributions plus
`n = 4095 ... 2097153`).

## Extra Credit 2: Shared-Memory Scan

`shared_scan.{h,cu}`: GPU Gems chapter 39, Examples 39-1 (Hillis-Steele) and 39-2
(Blelloch) in **dynamic shared memory**, extended hierarchically to any size. The
block size is the last argument (a power of two, default 256):

```cpp
#include <stream_compaction/shared_scan.h>

int input[N], output[N];
StreamCompaction::SharedScan::scanNaive(n, output, input);            // Example 39-1
StreamCompaction::SharedScan::scanEfficient(n, output, input);        // Example 39-2
StreamCompaction::SharedScan::scanEfficientPadded(n, output, input);  // 39-2, padded
StreamCompaction::SharedScan::scanNaive(n, output, input, 128);       // block size 128
```

Every scan is three steps:

1. Tile kernel (39-1 `kernTileScanNaive` or 39-2 `kernTileScanTree<PADDED>`): one block per tile loads its tile into dynamic shared memory, scans it, writes the exclusive result back and stores the tile total in `tileSums[block]`.
2. `Efficient::scanDevice` scans the tile totals.
3. `kernAddTileOffsets` adds each tile's offset to its elements.

At `2^22` with block 256 that is 11 launches instead of 25.

Medians of 20 x 3 samples, same random input as above:

| n | CPU | Efficient (fused, Part 5) | 39-1 shared | 39-2 shared | 39-2 padded | Thrust |
|---|---|---|---|---|---|---|
| 4,096 | 0.0021 | 0.0307 | **0.0205** | 0.0236 | 0.0236 | 0.0368 |
| 16,384 | 0.0057 | 0.0492 | **0.0236** | 0.0266 | 0.0266 | 0.0350 |
| 65,536 | 0.0222 | 0.1341 | **0.0571** | 0.0669 | 0.0654 | 0.0369 |
| 262,144 | 0.0906 | 0.1811 | **0.0727** | 0.1006 | 0.0973 | 0.3686 |
| 1,048,576 | 0.3797 | 0.4096 | **0.1341** | 0.2201 | 0.1980 | 0.3768 |
| 4,194,304 | 1.8405 | 1.3641 | **0.4124** | 0.7200 | 0.7011 | 0.5270 |

The shared-memory scan is 3.3x faster than the fused tree scan at 4M and also wins
at `2^12` (0.0205 vs 0.0307 ms), 1.28x faster than Thrust and 4.5x faster than the
serial CPU scan. **39-1 beats 39-2 at every size** (1.2-1.8x): its accesses are
all stride 1 and every thread is busy at every level, while the tree idles half
its threads per level.

### Bank conflicts and occupancy

The tree walks nodes at stride `2 * offset` (a 2-way conflict at `offset = 1`); the
padded variant stores element `i` at `i + i / 32`:

| layout (isolated probe: 4096 blocks x 256 threads x 200 repeats) | time (ms) | relative |
|---|---|---|
| Example 39-2, stride-2 accesses | 11.4166 | 1.00x |
| Example 39-2, one pad slot per 32 elements | 12.7980 | 1.12x |

Padding removes the conflicts by construction but does *not* pay off here: a 3%
win end to end (0.7011 vs 0.7200 ms at 4M) and 12% slower in the isolated probe,
i.e. the scan is not shared-memory-throughput bound and the algorithm matters
more. (Nsight Compute counters need administrator rights on this machine:
`ERR_NVGPUCTRPERM`, hence the direct measurement.)

| block @ `2^22` | 39-1 (ms) | 39-2 padded (ms) | shared bytes | tiles |
|---|---|---|---|---|
| 128 | 0.4124 | 0.6741 | 528 | 32768 |
| 256 | 0.3740 | 0.7065 | 1056 | 16384 |
| 512 | 0.4384 | 0.7820 | 2112 | 8192 |
| 1024 | 0.5763 | 1.1729 | 4224 | 4096 |

256 wins for 39-1 and 128 for the tree; 1024 is 1.5-1.7x slower (4224 B of shared
memory per block, only two blocks per SM, fewer and larger tiles for the tile-sum
scan) - the occupancy tradeoff the assignment points at. The tile kernels use 12/19
registers with no spills.

**Tests:** 18 checks (6 sizes x 3 variants: 256, 253, 33, 10000, 1, `2^20`, which
covers partial tiles, non-power-of-two sizes and a single tile) against the CPU
scan; a 4320-scan randomized sweep (240 sizes x 6 block sizes x 3 variants)
matched the reference exactly; compute-sanitizer reports 0 errors. This module
also exposed a latent bug: `Efficient::scanDevice(1, ...)` wrote `data[0] = 0` on
the host with a device pointer (a crash) - it now launches the existing one-thread
zeroing kernel.

## Test output

Output of the supplied test program (default `SIZE = 256`, Release build):

```
****************
** SCAN TESTS **
****************
    [  15   8  14  35   8  49  36  19  42  42  46  36  22 ...   0   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0003ms    (std::chrono Measured)
    [   0  15  23  37  72  80 129 165 184 226 268 314 350 ... 6316 6316 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0001ms    (std::chrono Measured)
    [   0  15  23  37  72  80 129 165 184 226 268 314 350 ... 6276 6305 ]
    passed 
==== naive scan, power-of-two ====
   elapsed time: 0.146432ms    (CUDA Measured)
    passed 
==== naive scan, non-power-of-two ====
   elapsed time: 0.07168ms    (CUDA Measured)
    passed 
==== work-efficient scan, power-of-two ====
   elapsed time: 0.098304ms    (CUDA Measured)
    passed 
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.0256ms    (CUDA Measured)
    passed 
==== thrust scan, power-of-two ====
   elapsed time: 0.103424ms    (CUDA Measured)
    passed 
==== thrust scan, non-power-of-two ====
   elapsed time: 0.03776ms    (CUDA Measured)
    passed 

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   2   0   1   2   1   2   3   0   0   0   0   2 ...   0   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0011ms    (std::chrono Measured)
    [   1   2   1   2   1   2   3   2   3   3   2   1   3 ...   3   3 ]
    passed 
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0003ms    (std::chrono Measured)
    [   1   2   1   2   1   2   3   2   3   3   2   1   3 ...   3   3 ]
    passed 
==== cpu compact with scan ====
   elapsed time: 0.0013ms    (std::chrono Measured)
    [   1   2   1   2   1   2   3   2   3   3   2   1   3 ...   3   3 ]
    passed 
==== work-efficient compact, power-of-two ====
   elapsed time: 0.089088ms    (CUDA Measured)
    passed 
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.043008ms    (CUDA Measured)
    passed 

**********************
** RADIX SORT TESTS **
**********************
    [   5  -3   0 128   5   7 -1000   2 ]
==== gpu radix sort, README example ====
   elapsed time: 0.530432ms    (CUDA Measured)
    [ -1000  -3   0   2   5   5   7 128 ]
    passed 
    [  65  58  64  85  58  49  86  19  92  92  96  36  22 ...   0   0 ]
==== gpu radix sort, power-of-two, values in [0, 100) ====
   elapsed time: 0.176128ms    (CUDA Measured)
    [   0   0   0   0   0   0   0   1   1   1   2   3   3 ...  98  99 ]
    passed 
==== gpu radix sort, non-power-of-two, values in [0, 100) ====
   elapsed time: 0.205824ms    (CUDA Measured)
    passed 
    [ -71 783 404  31 679 285 -237 -23 926 -423 949 778 -178 ... 765 -643 ]
==== gpu radix sort, signed values in [-1000, 1000) ====
   elapsed time: 0.180224ms    (CUDA Measured)
    [ -1000 -1000 -996 -993 -993 -992 -991 -988 -985 -979 -979 -973 -967 ... 996 997 ]
    passed 
==== gpu radix sort, all elements equal ====
   elapsed time: 0.196608ms    (CUDA Measured)
    passed 
==== gpu radix sort, already sorted input ====
   elapsed time: 0.200704ms    (CUDA Measured)
    passed 
==== gpu radix sort, reverse sorted input ====
   elapsed time: 0.15872ms    (CUDA Measured)
    passed 
==== gpu radix sort, INT_MIN / INT_MAX extremes ====
   elapsed time: 0.187392ms    (CUDA Measured)
    passed 
==== gpu radix sort, single element ====
   elapsed time: 0.155648ms    (CUDA Measured)
    [  65 ]
    passed 
==== gpu radix sort, 20000 equal elements (multi-block) ====
   elapsed time: 0.201728ms    (CUDA Measured)
    passed 
==== gpu radix sort, 20000 values with 3 distinct keys ====
   elapsed time: 0.23152ms    (CUDA Measured)
    passed 
==== gpu radix sort, 12345 signed values (multi-block, non-power-of-two) ====
   elapsed time: 0.2048ms    (CUDA Measured)
    passed 
==== gpu radix sort, 2^20 random 32-bit values ====
   elapsed time: 1.23475ms    (CUDA Measured)
    passed 

*******************************
** SHARED MEMORY SCAN TESTS **
*******************************
    [  15   8  14  35   8  49  36  19  42  42  46  36  22 ...   0   0 ]
==== shared-memory scan, power-of-two (256) ====
   elapsed time: 0.096256ms    (CUDA Measured) Example 39-1, Hillis-Steele in shared memory
    passed 
   elapsed time: 0.019456ms    (CUDA Measured) Example 39-2, tree in shared memory
    passed 
   elapsed time: 0.044096ms    (CUDA Measured) Example 39-2, padded shared layout
    passed 
==== shared-memory scan, non-power-of-two (253) ====
   elapsed time: 0.012288ms    (CUDA Measured) Example 39-1, Hillis-Steele in shared memory
    passed 
   elapsed time: 0.01536ms    (CUDA Measured) Example 39-2, tree in shared memory
    passed 
   elapsed time: 0.014336ms    (CUDA Measured) Example 39-2, padded shared layout
    passed 
==== shared-memory scan, 33 elements (single partial tile) ====
   elapsed time: 0.011264ms    (CUDA Measured) Example 39-1, Hillis-Steele in shared memory
    passed 
   elapsed time: 0.01536ms    (CUDA Measured) Example 39-2, tree in shared memory
    passed 
   elapsed time: 0.01536ms    (CUDA Measured) Example 39-2, padded shared layout
    passed 
==== shared-memory scan, 10000 elements (last tile partial) ====
   elapsed time: 0.023552ms    (CUDA Measured) Example 39-1, Hillis-Steele in shared memory
    passed 
   elapsed time: 0.0256ms    (CUDA Measured) Example 39-2, tree in shared memory
    passed 
   elapsed time: 0.0256ms    (CUDA Measured) Example 39-2, padded shared layout
    passed 
==== shared-memory scan, single element ====
   elapsed time: 0.012288ms    (CUDA Measured) Example 39-1, Hillis-Steele in shared memory
    passed 
   elapsed time: 0.01536ms    (CUDA Measured) Example 39-2, tree in shared memory
    passed 
   elapsed time: 0.014336ms    (CUDA Measured) Example 39-2, padded shared layout
    passed 
==== shared-memory scan, 2^20 elements ====
   elapsed time: 0.355008ms    (CUDA Measured) Example 39-1, Hillis-Steele in shared memory
    passed 
   elapsed time: 0.379776ms    (CUDA Measured) Example 39-2, tree in shared memory
    passed 
   elapsed time: 0.409152ms    (CUDA Measured) Example 39-2, padded shared layout
Press any key to continue . . . 
    passed 
```

## CMakeLists.txt changes

Both `CMakeLists.txt` files changed beyond their source lists:

* Root: added the `stream_compaction_bench` executable
  (`src/bench.cu`), with the same version-guarded `CUDA_ARCHITECTURES` and
  Windows-only `/Zc:preprocessor` as the other targets. Modes:
  `--sizes 12,14,16,18,20,22 --iters 50 --reps 3 --csv` (scans), `--sort` (radix
  sort), `--smem` (shared-memory scan).
* `stream_compaction/CMakeLists.txt`: fixed a stray `}` in
  `set_target_properties(stream_compaction} ...)` that breaks CMake < 3.23, and
  added the extra-credit modules to the file lists. Nothing else changed.
