// Copyright 2024 gQuery Contributors
#include <core/parallel.h>
#include <util/check.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <vector>
#include <numeric>

using namespace gquery;

class ParallelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize the parallel system with default thread count
        ParallelInit();
    }

    void TearDown() override {
        ParallelCleanup();
    }
};

TEST_F(ParallelTest, ParallelForSingleThread) {
    // Test with a single thread
    ParallelCleanup();
    ParallelInit(1);
    
    std::vector<int> values(1000, 0);
    
    ParallelFor(0, values.size(), [&](int64_t i) {
        values[i] = i;
    });
    
    for (int i = 0; i < values.size(); ++i) {
        EXPECT_EQ(values[i], i);
    }
}

TEST_F(ParallelTest, ParallelForMultiThread) {
    std::vector<int> values(10000, 0);
    
    ParallelFor(0, values.size(), [&](int64_t i) {
        values[i] = i;
    });
    
    for (int i = 0; i < values.size(); ++i) {
        EXPECT_EQ(values[i], i);
    }
}

TEST_F(ParallelTest, ParallelForEmptyRange) {
    // Test with empty range
    int counter = 0;
    ParallelFor(0, 0, [&](int64_t) { counter++; });
    
    EXPECT_EQ(counter, 0);
}

TEST_F(ParallelTest, ParallelForThreadSafety) {
    const int size = 100000;
    std::atomic<int> sum(0);
    
    // This test will have multiple threads updating the same atomic variable
    ParallelFor(0, size, [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            sum += 1;
        }
    });
    
    EXPECT_EQ(sum, size);
}

TEST_F(ParallelTest, ParallelForChunkSize) {
    const int size = 10000;
    std::vector<int> chunkSizes;
    std::mutex mutex;
    
    ParallelFor(0, size, [&](int64_t start, int64_t end) {
        std::lock_guard<std::mutex> lock(mutex);
        chunkSizes.push_back(end - start);
    });
    
    // Verify we got multiple chunks
    EXPECT_GT(chunkSizes.size(), 1);
    
    // Sum should equal the total size
    int totalProcessed = std::accumulate(chunkSizes.begin(), chunkSizes.end(), 0);
    EXPECT_EQ(totalProcessed, size);
}

TEST_F(ParallelTest, ThreadPoolFunctions) {
    int originalThreads = RunningThreads();
    EXPECT_GT(originalThreads, 0);
    
    // Test disable/reenable
    DisableThreadPool();
    ReenableThreadPool();
    
    // Make sure count is still the same
    EXPECT_EQ(RunningThreads(), originalThreads);
}

// Test to ensure we get near-linear speedup with multiple threads
TEST_F(ParallelTest, PerformanceScaling) {
    // Skip in debug builds as timing will be inconsistent
    #ifndef NDEBUG
    GTEST_SKIP() << "Skipping performance test in debug build";
    #endif
    
    const int size = 10000000;
    std::vector<double> values(size);
    
    // First measure single-threaded time
    ParallelCleanup();
    ParallelInit(1);
    
    auto start1 = std::chrono::high_resolution_clock::now();
    ParallelFor(0, size, [&](int64_t i) {
        values[i] = std::sin(static_cast<double>(i) * 0.0001) * std::cos(static_cast<double>(i) * 0.0002);
    });
    auto end1 = std::chrono::high_resolution_clock::now();
    auto singleThreadTime = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    
    // Now measure multi-threaded time
    ParallelCleanup();
    ParallelInit(); // Default to hardware threads
    
    auto start2 = std::chrono::high_resolution_clock::now();
    ParallelFor(0, size, [&](int64_t i) {
        values[i] = std::sin(static_cast<double>(i) * 0.0001) * std::cos(static_cast<double>(i) * 0.0002);
    });
    auto end2 = std::chrono::high_resolution_clock::now();
    auto multiThreadTime = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
    
    // We should see at least some speedup, though exact amount will vary by system
    EXPECT_LT(multiThreadTime, singleThreadTime);
    
    // Print actual results
    std::cout << "Single thread time: " << singleThreadTime << "ms" << std::endl;
    std::cout << "Multi thread time: " << multiThreadTime << "ms with " 
              << RunningThreads() << " threads" << std::endl;
    std::cout << "Speedup factor: " << static_cast<double>(singleThreadTime) / multiThreadTime << std::endl;
} 