# CUDA Stream Compaction

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 2 - Stream Compaction**

* **Jing Huang**
  * [GitHub](https://github.com/Stabil1ze)
* Tested on: Windows 11, Intel i7-12700H @ 2.30GHz 23GB,
  NVIDIA GeForce RTX 3060 Laptop GPU 6GB (Personal computer)

## Overview

This project implements several versions of the **scan** (exclusive prefix sum)
algorithm together with **stream compaction** that removes all `0`s from an
array of `int`s. The same building blocks (scan + scatter) will later be used
in the path tracer to compact away terminated rays.

Implemented features:

1. **CPU scan & compaction** (`cpu.cu`)
   - Serial exclusive scan (reference implementation for every GPU test).
   - `compactWithoutScan`: one pass with a running output pointer.
   - `compactWithScan`: CPU version of map -> scan -> scatter.
2. **Naive GPU scan** (`naive.cu`)
   - GPU Gems 3, 39.2.1 style scan using two global-memory buffers that are
     swapped for each doubling offset (`ilog2ceil(n)` kernel invocations).
   - One block size for the whole grid (512, see below).
3. **Work-efficient GPU scan & compaction** (`efficient.cu`, `common.cu`)
   - Blelloch up-sweep / down-sweep. Only the still-active tree nodes are
     launched at each level, and the node index is derived from the level's
     active merge count, so no thread ever touches a node that does not exist.
   - **Part 5 (extra credit)**: the upper tree levels - every level whose whole
     merge count fits in one block - are *fused* into a single 1024-thread
     kernel that walks those levels with `__syncthreads()` in between. This
     drops the launch count per scan from `2*log2(m)+1` to
     `2*log2(m/2048)+3` (45 -> 25 launches at `n = 2^22`, 25 -> 5 at
     `n = 2^12`), which pulls the work-efficient scan level with the serial CPU
     scan at about one million elements and ahead of it beyond that. Details in
     [Part 5](#part-5-extra-credit-5-why-is-my-work-efficient-gpu-scan-slower-than-the-cpu)
   - `Common::kernMapToBoolean` and `Common::kernScatter` implement stream
     compaction on top of the scan.
4. **Thrust scan** (`thrust.cu`)
   - Thin wrapper around `thrust::exclusive_scan`; device-vector setup and the
     final copy are excluded from the measured region.
5. **CUDA error checking** (`naive.cu`, `efficient.cu`, `thrust.cu`)
   - Every `cudaMalloc`, `cudaMemcpy` and kernel launch is followed by
     `checkCUDAError(...)` (11 call sites), always placed *outside* the timed
     regions so the measured times stay comparable with the reference harness.
6. **Radix sort on top of the scan** (extra credit 1, `radix_sort.{h,cu}`)
   - Stable 8-bit LSD radix sort: per-block histograms, the work-efficient scan
     for the global bucket offsets, then a warp-ranked stable scatter. Four
     passes sort a whole 32-bit `int`, negative values included. See
     [Extra Credit 1](#extra-credit-1-radix-sort-10).

All GPU scans support **non-power-of-two** inputs: work-efficient kernels pad
the logical array to the next power of two and only report the first `n`
results.

## Performance Analysis

All numbers below come from `src/bench.cu`, a small benchmark harness added for
this write-up (see [CMakeLists.txt changes](#cmakeliststxt-changes)). It drives
the implementations through the project's own `PerformanceTimer`, so the timed
region is exactly the one of the supplied test program: kernel work only, with
`cudaMalloc`/`cudaMemcpy` excluded. For every size the harness collects
`50 x 3 = 150` samples per implementation and reports their median; each cell
of the table is the median of three such processes. Arrays are randomly filled
with values in `[0, 100)`. The harness also checks every variant against the
CPU reference on every size and exits non-zero on mismatch.

Release build (CMake + Ninja, CUDA 13.3), RTX 3060 Laptop GPU.

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

I swept the block size of each GPU scan at `n = 2^22` (median of many runs) and
kept the best value:

| Implementation | Sizes tried | Best | Notes |
|---|---|---|---|
| Naive scan | 128, 256, 512, 1024 | 512 | Memory-bound full-grid kernels; 512 balances blocks/SM with coalesced global access |
| Work-efficient, per-level kernels | 64, 128, 256 | 64 | Deep tree levels launch very few threads; small blocks reduce launch idle work without hurting occupancy |
| Work-efficient, Part 5 fused block | 256, 512, 1024 | 1024 | The fused block should cover as many levels as possible, so the largest legal block wins |

The fused block size is the only one that changes the *number of kernel
launches*, so it is the only one that matters much:

| `FUSED_BLOCK_SIZE` | launches @ `2^12` | time @ `2^12` (ms) | launches @ `2^22` | time @ `2^22` (ms) |
|---|---|---|---|---|
| 256 | 9 | 0.0508 | 29 | 1.341 |
| 512 | 7 | 0.0328 | 27 | 1.306 |
| 1024 | 5 | 0.0272 | 25 | 1.329 |

(Each cell is a `50 x 3` median; the 512 configuration was measured once, the
others twice. At `2^22` the three configurations are within run-to-run noise;
at `2^12` each extra launch costs ~5 us, which is exactly the ~0.02 ms
spread between the 5-launch and the 9-launch configuration.)

### Observations

* **The serial CPU scan wins up to about `n = 2^20` and loses above it.** At
  `n = 2^12` the CPU needs 0.0015 ms while the fastest GPU scan needs
  0.0236 ms, but the crossing point is reached at `n = 2^20` (CPU 0.3634 ms vs
  work-efficient 0.3793 ms - a tie - and Thrust 0.3459 ms), and at `n = 2^22`
  the work-efficient scan is 1.35x and Thrust 3.3x faster than the CPU. The
  CPU loop is a straight O(n) pass over cache-friendly memory with zero setup
  cost; a GPU scan pays dozens of micro seconds of fixed launch/issue cost
  before it has moved a single byte.
* **The naive scan scales the worst.** It is an O(n log n) algorithm: each of
  its 22 doubling kernels at `n = 2^22` sweeps the whole array through global
  memory (about 34 MB per kernel, ~740 MB in total, plus the final shift pass).
  Its bottleneck is pure memory I/O, and it is the slowest implementation
  at 4M elements (2.8851 ms).
* **The work-efficient scan moves far less data** - only the active tree nodes,
  about 100 MB in total at `n = 2^22` - but it pays one kernel launch per level.
  Before Part 5 that overhead made it slower than the CPU up to ~2^21 elements;
  after fusing the upper levels it is the fastest hand-written implementation
  here at `n = 2^22` (1.2528 ms vs naive 2.8851 ms).
* **Thrust is the fastest implementation for `n >= 2^18`.** CUB's device-wide
  scan is a single kernel using a decoupled look-back design with high
  occupancy and one pass over memory; it needs 0.5180 ms at `n = 2^22`, 2.4x
  faster than my best tree scan. Below ~64K elements its own fixed dispatch cost
  dominates, so it hovers around 0.03 ms regardless of size and even gets
  *slower* again at `2^18`-`2^20` (0.32-0.35 ms), which is where the internal
  temporary-buffer management and the fallback to a multi-pass path show up.
* **Block size is a second-order effect** (10-30%) for the naive and per-level
  work-efficient scans. Both are dominated at small `n` by launch latency and
  at large `n` by global-memory traffic, not by per-thread instruction count.

## Part 5 (extra credit, +5): why is my work-efficient GPU scan slower than the CPU?

### The symptom

Before Part 5 my work-efficient scan needed **0.1423 ms at `n = 2^12`**, while
the serial CPU scan needed **0.0015 ms** - roughly 95x slower - and it stayed
behind the CPU until roughly `2^21` elements. The hints in Part 5 point at the
lazy threads of the upper tree levels; measuring them one by one showed that
this is only half of the story.

### Where the time goes

There are two candidate costs, and the harness measures both separately
(`Lazy` = full `m/2`-thread grid on every level with an early-exit guard,
`PerLevel` = only `ceil(active/64)` blocks per level, `Efficient` = the fused
version of this project):

| n | Lazy (ms) | Per-level (ms) | Fused (ms) |
|---|---|---|---|
| 4,096 | 0.1454 | 0.1423 | 0.0236 |
| 16,384 | 0.1587 | 0.1577 | 0.0379 |
| 65,536 | 0.1994 | 0.1889 | 0.1163 |
| 262,144 | 0.2686 | 0.2372 | 0.1420 |
| 1,048,576 | 0.6877 | 0.4651 | 0.3793 |
| 4,194,304 | 2.5238 | 1.3532 | 1.2528 |

1. **The number of kernel launches dominates at small `n`.** A level-by-level
   Blelloch scan issues `2*log2(m)+1` kernels (up-sweep, one kernel that zeroes
   the last element, down-sweep): 25 launches at `2^12` and 45 at `2^22`. The
   harness measures the back-to-back cost of an empty kernel launch as
   **6.9 us** (5.4 - 8.9 us across processes, depending on clocks). 25 such
   launches at ~6 us each account for essentially all of the 0.1423 ms measured
   at `2^12`; the tree work itself is a handful of microseconds.
2. **Compacting the threads alone does not help.** `Lazy` and `PerLevel` differ
   by 2% at `2^12` (0.1454 vs 0.1423 ms) - within noise - because a kernel with
   one block costs exactly as much to launch as a kernel with 32,768 blocks. So
   the "some threads are lazy, terminate them early" hint by itself buys
   nothing; it only pays off once the launch count is under control. At
   `n = 2^22` the lazy version does become 1.9x slower (2.5238 vs 1.3532 ms),
   because there it has to *dispatch* 32,768 idle blocks per level.

To check the launch-overhead explanation quantitatively, compare the fused
version with the per-level version launch by launch:

| n | launches before | launches after | removed | time before (ms) | time after (ms) | speedup | implied cost per removed launch |
|---|---|---|---|---|---|---|---|
| 4,096 | 25 | 5 | 20 | 0.1423 | 0.0236 | **6.0x** | 5.9 us |
| 16,384 | 29 | 9 | 20 | 0.1577 | 0.0379 | **4.2x** | 6.0 us |
| 65,536 | 33 | 13 | 20 | 0.1889 | 0.1163 | 1.6x | 3.6 us |
| 262,144 | 37 | 17 | 20 | 0.2372 | 0.1420 | 1.7x | 4.8 us |
| 1,048,576 | 41 | 21 | 20 | 0.4651 | 0.3793 | 1.2x | 4.3 us |
| 4,194,304 | 45 | 25 | 20 | 1.3532 | 1.2528 | 1.08x | 5.0 us |

The last column is `(time before - time after) / 20`, and it lands in the
3.6 - 6.0 us range at *every* size, close to the directly measured 6.9 us of an
empty launch. In other words: **the gain is exactly "removed launches x launch
latency"** - nothing else in the measured time changed, which is what you would
expect if the bottleneck really is kernel issue and not arithmetic.

### The optimization: fuse the upper tree levels into one block

Once a level has at most one block of active merges, it does not need its own
launch any more: the same block can simply keep looping over the remaining
levels, with `__syncthreads()` between them so that the read-after-write
dependencies of the tree are respected.

```cpp
const int FUSED_BLOCK_SIZE = 1024;

// Up-sweep of every level whose merge count is <= FUSED_BLOCK_SIZE.
// The old offsets 1, 2, 4, ... are walked inside a single block.
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

// Down-sweep of the levels above `lastOffset`, in the same style.
__global__ void kernEfficientDownSweepFused(int m, int lastOffset, int *data) {
    for (int offset = m / 2; offset >= lastOffset; offset >>= 1) {
        int active = m / (2 * offset);
        if (threadIdx.x < active) {
            int node0 = (2 * threadIdx.x + 1) * offset - 1;
            int node1 = node0 + offset;
            data[node1] += data[node0];
            data[node0] = data[node1] - data[node0];
        }
        __syncthreads();
    }
}
```

`scanDevice` now splits the tree at `fusedFirst = m / (2 * FUSED_BLOCK_SIZE)`
(or `1` if that would be zero): everything below it stays a per-level launch
with the tuned block size, everything above it - the levels that used to run
with 1 to 1024 active threads - runs in one block.

```cpp
const int fusedFirst = (m > 2 * FUSED_BLOCK_SIZE) ? (m / (2 * FUSED_BLOCK_SIZE)) : 1;

for (int offset = 1; offset < fusedFirst; offset <<= 1) {      // bulk levels
    ...kernEfficientUpSweep<<<blocks, BLOCK_SIZE>>>(m, offset, data);
}
kernEfficientUpSweepFused<<<1, FUSED_BLOCK_SIZE>>>(m, fusedFirst, data);  // fused top

kernEfficientSetLastZero<<<1, 1>>>(m, data);

kernEfficientDownSweepFused<<<1, FUSED_BLOCK_SIZE>>>(m, fusedFirst, data); // fused top
for (int offset = fusedFirst >> 1; offset > 0; offset >>= 1) { // bulk levels
    ...kernEfficientDownSweep<<<blocks, BLOCK_SIZE>>>(m, offset, data);
}
```

(`fusedFirst` also drives two guards in the real code - `fusedFirst < m` before
the fused up-sweep and `fusedFirst <= m / 2` before the fused down-sweep - so
that the `m = 1` and `m = 2` cases stay correct.)

For `m <= 2048` the fused kernels cover the whole tree, so a complete scan is
three launches: up-sweep, zero the last element, down-sweep. For `m = 2^22` the
25 remaining launches are 11 bulk up-sweep levels, 1 fused up-sweep, the
zeroing, 1 fused down-sweep and 11 bulk down-sweep levels.

![Part 5: fusion and launch counts](img/performance-part5.png)

### Why the win shrinks with n, and what is left on the table

* The launches that disappear are the *cheap* ones. The bulk levels keep
  thousands of blocks of real work; not a single byte of memory traffic is
  removed, only launch overhead. At `n = 2^22` that overhead is ~5 us against a
  ~1.25 ms total, so a 1.08x speedup is all the structure allows.
* At large `n` the GPU stays busy during the bulk phase, and the launches of
  neighbouring levels partly overlap with the memory traffic of the previous
  kernel, so the effective launch cost is lower than measured in isolation. At
  small `n` every launch is exposed.
* Levels executed sequentially inside one block cannot overlap with the next
  kernel the way back-to-back launches can, so a small part of the saved
  overhead is paid back.
* Going further would require removing launches instead of merging them: a
  shared-memory tile scan with one block per tile plus a scan of the tile sums
  (Extra Credit 2), or a single-pass decoupled look-back scan (what Thrust/CUB
  does - 0.5180 ms at `n = 2^22`, still 2.4x faster than my best tree scan).

### Correctness of the fused version

* `src/bench.cu` verifies Naive, Efficient, Thrust, Lazy and PerLevel against
  the CPU reference at every benchmarked size (`VERIFY FAIL` + non-zero exit on
  mismatch). All clean.
* The supplied scan/compaction tests pass 12/12 (power-of-two and
  non-power-of-two), both with the default `SIZE = 256` and with
  `SIZE = 1 << 20`; the 13 radix sort tests added for extra credit 1 are
  described below.
* `compute-sanitizer --tool memcheck` over the fused *and* the pre-fusion/lazy
  variants at `n = 2^12` and `n = 2^20`: `ERROR SUMMARY: 0 errors`.
* Integer overflow is a real trap at the top level. The node index
  `(index + 1) * (2 * offset) - 1` is computed in `int`; at `m = 2^22` the
  product reaches `2^31` for the threads that no longer have a node. Guarding
  only the *store* was not enough while I was writing the full-grid variant -
  `compute-sanitizer` still reported invalid reads because the address
  computation had been predicated rather than branched away. The kernels here
  therefore compute `active = m / (2 * offset)` first and only compute the
  index for threads that pass `index < active`.
* CUDA failures are no longer silent: with the `checkCUDAError` calls added in
  this revision, running the test program with `CUDA_VISIBLE_DEVICES=999`
  prints
  `CUDA error (naive.cu:49): Naive::scan: cudaMalloc / cudaMemcpy(H2D) failed: no CUDA-capable device is detected`
  and exits with status 1.

## Extra Credit 1: Radix Sort (+10)

`stream_compaction/radix_sort.{h,cu}` adds a GPU radix sort that is built on the
work-efficient scan of Part 3. It is a **stable LSD radix sort with 8-bit
digits**, so four counting passes sort a whole 32-bit `int`.

```cpp
#include <stream_compaction/radix_sort.h>

int input[8]  = { 5, -3, 0, 128, 5, 7, -1000, 2 };
int output[8];
StreamCompaction::RadixSort::sort(8, output, input);
```

Output of that call (also run as the first test case below, see
`src/main.cpp`):

```
    [   5  -3   0 128   5   7 -1000   2 ]      <- input
    [ -1000  -3   0   2   5   5   7 128 ]      <- output
```

### How it works

One pass per digit, least significant byte first:

1. `kernRadixHistogram` - every block builds a 256-bin histogram of its 4096
   elements in shared memory (one warp per contiguous 512-element chunk) and
   writes the block totals to global memory in **bin-major** layout,
   `hist[digit * numBlocks + block]`.
2. `Efficient::scanDevice(histPadded, devHist)` - the exclusive scan of that
   array *is* the start offset of every (digit, block) pair: in bin-major layout
   the prefix at `(digit, block)` counts all elements with a smaller digit plus
   the elements of the same digit in the preceding blocks.
3. `kernRadixScatter` - every element computes its stable rank inside its warp
   with a warp-level shuffle ranking, adds the per-warp offset and the scanned
   block offset, and writes itself to that position.

Every pass is stable, because blocks, the warps inside a block and the lanes
inside a warp are all processed in input order; that is what makes the four LSD
passes produce a fully sorted array. Negative values are handled by flipping the
sign bit before the digits are extracted, so the unsigned digit order equals the
order of the signed ints.

`Efficient::scanDevice` was added to `efficient.h` for this: it runs the
existing work-efficient scan in place on a device buffer, so the per-pass
histogram scans stay in device memory instead of round-tripping through the
host. The 256-bin histogram is padded to the next power of two (the padding is
never written by the histogram kernel and never read by the scatter).

### Performance

Release build, RTX 3060 Laptop GPU, arrays of mixed-sign random values (every
third element is negated). The GPU time covers the four passes only;
allocations and copies are outside the timed region, as everywhere else in this
project. `std::sort` is sampled 5 times and the GPU sorts 20 times per size
(median), because a single `std::sort` of 4M ints already takes ~0.1 s.

| n | std::sort (ms) | RadixSort (ms) | thrust::sort (ms) | speedup vs std::sort |
|---|---|---|---|---|
| 4,096 | 0.1129 | 0.1772 | 0.0594 | 0.64x |
| 16,384 | 0.4604 | 0.2222 | 0.1640 | 2.1x |
| 65,536 | 1.6824 | 0.3132 | 0.1208 | 5.4x |
| 262,144 | 6.8671 | 0.4188 | 0.4864 | 16.4x |
| 1,048,576 | 28.0361 | 0.9173 | 0.7680 | 30.6x |
| 4,194,304 | 113.5395 | 2.3994 | 1.8555 | 47.3x |

(Each cell is a single-process median; repeated runs spread by about 10-30% at the
GPU sizes below 262K elements, so the small-
 columns are representative
rather than exact. The radix kernels use 8 KB of shared memory and 26/40
registers with no spills, which allows 6 blocks / 1536 threads per SM.)

The same launch-overhead story as in Part 5 shows up at the small end: at
`n = 2^12` the sort issues 20 kernel launches (4 passes x [histogram, the
3-launch scan of a 256-entry histogram, scatter]), which at ~6 us per launch is
~0.12 ms of the 0.177 ms measured - more than the CPU needs for the whole sort.
Above ~16K elements the O(n) counting passes win, and at 4M elements the sort is
47x faster than `std::sort`. Thrust's `thrust::sort` (CUB's single-pass
segmented radix sort) is still 1.3x faster than this implementation because it
needs one kernel per pass instead of a histogram / scan / scatter pipeline with
a global round trip for the offsets.

Reproduce with `stream_compaction_bench.exe --sort`.

### Correctness tests

`src/main.cpp` has a `RADIX SORT TESTS` section that runs **13 cases**, each
compared element by element against `std::sort` on the CPU (the sorted sequence
of an int array is unique, so this covers duplicate keys as well):

* the 8-element README example above,
* 256 random values in `[0, 100)` (power-of-two) and the same data truncated to
  253 elements (non-power-of-two),
* 1024 signed values in `[-1000, 1000)`,
* all elements equal, already sorted input, reverse sorted input,
* `INT_MIN` / `INT_MAX` / `0` / `-1` extremes,
* a single element,
* 20000 equal elements and 20000 values with 3 distinct keys (5 blocks: the
  cross-block offsets of the histogram have to line up),
* 12345 signed values (multi-block *and* non-power-of-two, so the histogram
  padding is used),
* `2^20` values covering the whole 32-bit range (xorshift).

All 13 pass, and `compute-sanitizer --tool memcheck` over the complete test
program reports `ERROR SUMMARY: 0 errors`. As an extra check I also ran a
temporary out-of-tree fuzz harness against the same `std::sort` reference:
300 random cases with `n` in `[1, 60000]` over six key distributions (full 32-bit
range, a few distinct keys, mostly zeros, already sorted, reverse sorted,
non-negative noise), plus 35 cases at `n = 4095 ... 2097153` including
all-equal and two-key arrays - 335/335 matched `std::sort`.
## Test output

Output of the supplied test program (default `SIZE = 256`, Release build):

```
****************
** SCAN TESTS **
****************
    [  27   8  49   8   3  22  40  19   8  44  31   2   8 ...  29   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.0003ms    (std::chrono Measured)
    [   0  27  35  84  92  95 117 157 176 184 228 259 261 ... 6458 6487 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.0001ms    (std::chrono Measured)
    [   0  27  35  84  92  95 117 157 176 184 228 259 261 ... 6380 6384 ]
    passed 
==== naive scan, power-of-two ====
   elapsed time: 0.259072ms    (CUDA Measured)
    passed 
==== naive scan, non-power-of-two ====
   elapsed time: 0.067584ms    (CUDA Measured)
    passed 
==== work-efficient scan, power-of-two ====
   elapsed time: 0.763904ms    (CUDA Measured)
    passed 
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.018432ms    (CUDA Measured)
    passed 
==== thrust scan, power-of-two ====
   elapsed time: 0.096256ms    (CUDA Measured)
    passed 
==== thrust scan, non-power-of-two ====
   elapsed time: 0.047328ms    (CUDA Measured)
    passed 

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   3   0   1   0   1   0   2   1   2   2   3   0   2 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.0008ms    (std::chrono Measured)
    [   3   1   1   2   1   2   2   3   2   2   3   2   3 ...   3   1 ]
    passed 
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.0004ms    (std::chrono Measured)
    [   3   1   1   2   1   2   2   3   2   2   3   2   3 ...   2   1 ]
    passed 
==== cpu compact with scan ====
   elapsed time: 0.0011ms    (std::chrono Measured)
    [   3   1   1   2   1   2   2   3   2   2   3   2   3 ...   3   1 ]
    passed 
==== work-efficient compact, power-of-two ====
   elapsed time: 0.18432ms    (CUDA Measured)
    passed 
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.050176ms    (CUDA Measured)
    passed 

**********************
** RADIX SORT TESTS **
**********************
    [   5  -3   0 128   5   7 -1000   2 ]
==== gpu radix sort, README example ====
   elapsed time: 0.761856ms    (CUDA Measured)
    [ -1000  -3   0   2   5   5   7 128 ]
    passed 
    [  27   8  49   8  53  72  90  69  58  94  31  52  58 ...  29   0 ]
==== gpu radix sort, power-of-two, values in [0, 100) ====
   elapsed time: 0.186368ms    (CUDA Measured)
    [   0   0   0   1   1   1   2   3   3   3   3   3   4 ...  99  99 ]
    passed 
==== gpu radix sort, non-power-of-two, values in [0, 100) ====
   elapsed time: 0.193536ms    (CUDA Measured)
    passed 
    [  65  56 -286 -320 566 389 -117 -78 -109 127 -341 217 -917 ... -641 -958 ]
==== gpu radix sort, signed values in [-1000, 1000) ====
   elapsed time: 0.191488ms    (CUDA Measured)
    [ -998 -995 -995 -995 -994 -992 -990 -989 -989 -987 -985 -985 -979 ... 997 997 ]
    passed 
==== gpu radix sort, all elements equal ====
   elapsed time: 0.18432ms    (CUDA Measured)
    passed 
==== gpu radix sort, already sorted input ====
   elapsed time: 0.193536ms    (CUDA Measured)
    passed 
==== gpu radix sort, reverse sorted input ====
   elapsed time: 0.177152ms    (CUDA Measured)
    passed 
==== gpu radix sort, INT_MIN / INT_MAX extremes ====
   elapsed time: 0.164864ms    (CUDA Measured)
    passed 
==== gpu radix sort, single element ====
   elapsed time: 0.165888ms    (CUDA Measured)
    [  27 ]
    passed 
==== gpu radix sort, 20000 equal elements (multi-block) ====
   elapsed time: 0.177152ms    (CUDA Measured)
    passed 
==== gpu radix sort, 20000 values with 3 distinct keys ====
   elapsed time: 0.22016ms    (CUDA Measured)
    passed 
==== gpu radix sort, 12345 signed values (multi-block, non-power-of-two) ====
   elapsed time: 0.232448ms    (CUDA Measured)
    passed 
==== gpu radix sort, 2^20 random 32-bit values ====
   elapsed time: 1.28685ms    (CUDA Measured)
Press any key to continue . . . 
    passed 
```
## CMakeLists.txt changes

The root `CMakeLists.txt` was modified beyond the `SOURCE_FILES` list:

* Added a second executable target, `stream_compaction_bench` (`src/bench.cu`,
  linked against `stream_compaction`). It produces every number in this
  write-up and re-checks all scan variants against the CPU reference at every
  size, so the Part 5 measurements can be reproduced with
  `stream_compaction_bench.exe --sizes 12,14,16,18,20,22 --iters 50 --reps 3 --csv`.
  `CUDA_ARCHITECTURES` (version-guarded exactly like the library target) and
  the Windows-only `/Zc:preprocessor` option mirror the other targets.
* Fixed a one-character typo in `stream_compaction/CMakeLists.txt`:
  `set_target_properties(stream_compaction} ...)` -> `set_target_properties(stream_compaction ...)`.
  With CMake < 3.23 that branch is taken, and the stray `}` makes CMake fail
  with "can not find target". Nothing else in that file was changed.
* The file lists of `stream_compaction/CMakeLists.txt` also contain
  `radix_sort.cu` / `radix_sort.h` now (extra credit 1); apart from that,
  the library's CMake file and the test target are untouched.