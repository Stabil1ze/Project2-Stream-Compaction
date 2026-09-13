/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <algorithm>
#include <climits>
#include <cstdio>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/radix_sort.h>
#include <stream_compaction/thrust.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 8; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

/**
 * Extra credit 1 helper: sorts a copy of `input` with std::sort and checks the
 * GPU radix sort against it. The sorted sequence of an int array is unique, so
 * a value-by-value comparison is a complete correctness check (duplicates
 * included).
 */
void testRadixSort(const char *desc, int n, const int *input, bool printValues) {
    int *expected = new int[n];
    int *actual = new int[n];

    std::copy(input, input + n, expected);
    std::sort(expected, expected + n);

    zeroArray(n, actual);
    printDesc(desc);
    StreamCompaction::RadixSort::sort(n, actual, input);
    printElapsedTime(StreamCompaction::RadixSort::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    if (printValues) {
        printArray(n, actual, true);
    }
    printCmpResult(n, expected, actual);

    delete[] expected;
    delete[] actual;
}

int main(int argc, char* argv[]) {
    // Scan tests

    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    /* For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    onesArray(SIZE, c);
    printDesc("1s array for finding bugs");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printArray(SIZE, c, true); */

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    printf("\n");
    printf("**********************\n");
    printf("** RADIX SORT TESTS **\n");
    printf("**********************\n");

    // Extra credit 1: every case below is checked element by element against
    // std::sort on the CPU.
    // Small hand-checkable example, also used in the README.
    {
        const int n = 8;
        int example[n] = { 5, -3, 0, 128, 5, 7, -1000, 2 };
        printArray(n, example, false);
        testRadixSort("gpu radix sort, README example", n, example, true);
    }

    genArray(SIZE - 1, a, 100);
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);
    testRadixSort("gpu radix sort, power-of-two, values in [0, 100)", SIZE, a, true);
    testRadixSort("gpu radix sort, non-power-of-two, values in [0, 100)", NPOT, a, false);

    {
        // Signed values: the sort flips the sign bit, so the negatives have to
        // come out in front of the non-negative values.
        const int n = 1024;
        int *signedInput = new int[n];
        for (int i = 0; i < n; ++i) {
            signedInput[i] = (int)(rand() % 2001) - 1000;
        }
        printArray(n, signedInput, true);
        testRadixSort("gpu radix sort, signed values in [-1000, 1000)", n, signedInput, true);
        delete[] signedInput;
    }

    {
        const int n = 1000;
        int *edge = new int[n];

        for (int i = 0; i < n; ++i) {
            edge[i] = 42;
        }
        testRadixSort("gpu radix sort, all elements equal", n, edge, false);

        for (int i = 0; i < n; ++i) {
            edge[i] = i - n / 2;
        }
        testRadixSort("gpu radix sort, already sorted input", n, edge, false);

        for (int i = 0; i < n; ++i) {
            edge[i] = n / 2 - i;
        }
        testRadixSort("gpu radix sort, reverse sorted input", n, edge, false);

        edge[0] = INT_MIN;
        edge[1] = INT_MAX;
        edge[2] = -1;
        edge[3] = 0;
        for (int i = 4; i < n; ++i) {
            edge[i] = (i % 2) ? INT_MIN : INT_MAX;
        }
        testRadixSort("gpu radix sort, INT_MIN / INT_MAX extremes", n, edge, false);

        delete[] edge;
    }

    testRadixSort("gpu radix sort, single element", 1, a, true);

    {
        // Degenerate key distributions spread over several blocks: the
        // per-block histogram offsets have to line up across blocks here.
        const int n = 20000;  // 5 blocks of 4096 elements
        int *many = new int[n];
        for (int i = 0; i < n; ++i) {
            many[i] = 7;
        }
        testRadixSort("gpu radix sort, 20000 equal elements (multi-block)", n, many, false);

        for (int i = 0; i < n; ++i) {
            many[i] = i % 3;
        }
        testRadixSort("gpu radix sort, 20000 values with 3 distinct keys", n, many, false);

        const int npot = 12345;
        for (int i = 0; i < npot; ++i) {
            many[i] = (int)(rand() % 1000) - 500;
        }
        testRadixSort("gpu radix sort, 12345 signed values (multi-block, non-power-of-two)", npot, many, false);
        delete[] many;
    }

    {
        // 1M values covering the whole 32-bit range (xorshift, so the sign bit
        // is exercised as well).
        const int n = 1 << 20;
        int *bigInput = new int[n];
        unsigned state = 0x9E3779B9u;
        for (int i = 0; i < n; ++i) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            bigInput[i] = (int)state;
        }
        testRadixSort("gpu radix sort, 2^20 random 32-bit values", n, bigInput, false);
        delete[] bigInput;
    }
    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}
