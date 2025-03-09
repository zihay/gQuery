// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <core/parallel.h>
#include <util/check.h>
#include <algorithm>
#include <iterator>
#include <list>
#include <thread>
#include <vector>

namespace gquery {

// Barrier Method Definitions
bool Barrier::Block() {
    std::unique_lock<std::mutex> lock(mutex);

    --numToBlock;
    if (numToBlock < 0) {
        throw std::runtime_error("Barrier::Block: numToBlock is negative");
    }

    if (numToBlock > 0) {
        cv.wait(lock, [this]() { return numToBlock == 0; });
    } else
        cv.notify_all();

    return --numToExit == 0;
}

ThreadPool *ParallelJob::threadPool;

// ThreadPool Method Definitions
ThreadPool::ThreadPool(int nThreads) {
    for (int i = 0; i < nThreads - 1; ++i)
        threads.push_back(std::thread(&ThreadPool::Worker, this));
}

void ThreadPool::Worker() {
    // Log message removed since we don't have that dependency
    std::unique_lock<std::mutex> lock(mutex);
    while (!shutdownThreads)
        WorkOrWait(&lock, false);
}

std::unique_lock<std::mutex> ThreadPool::AddToJobList(ParallelJob *job) {
    std::unique_lock<std::mutex> lock(mutex);
    // Add _job_ to head of _jobList_
    if (jobList)
        jobList->prev = job;
    job->next = jobList;
    jobList = job;

    jobListCondition.notify_all();
    return lock;
}

void ThreadPool::WorkOrWait(std::unique_lock<std::mutex> *lock, bool isEnqueuingThread) {
    if (!lock->owns_lock()) {
        throw std::runtime_error("WorkOrWait called without lock");
    }
    
    // Return if this is a worker thread and the thread pool is disabled
    if (!isEnqueuingThread && disabled) {
        jobListCondition.wait(*lock);
        return;
    }

    ParallelJob *job = jobList;
    while (job && !job->HaveWork())
        job = job->next;
    if (job) {
        // Execute work for _job_
        job->activeWorkers++;
        job->RunStep(lock);
        // Handle post-job-execution details
        if (!lock->owns_lock()) {
            lock->lock();
        }
        job->activeWorkers--;
        if (job->Finished())
            jobListCondition.notify_all();

    } else
        // Wait for new work to arrive or the job to finish
        jobListCondition.wait(*lock);
}

void ThreadPool::RemoveFromJobList(ParallelJob *job) {
    if (job->removed) {
        throw std::runtime_error("Job already removed from list");
    }

    if (job->prev)
        job->prev->next = job->next;
    else {
        if (jobList != job) {
            throw std::runtime_error("Job not at head of list as expected");
        }
        jobList = job->next;
    }
    if (job->next)
        job->next->prev = job->prev;

    job->removed = true;
}

bool ThreadPool::WorkOrReturn() {
    std::unique_lock<std::mutex> lock(mutex);

    ParallelJob *job = jobList;
    while (job && !job->HaveWork())
        job = job->next;
    if (!job)
        return false;

    // Execute work for _job_
    job->activeWorkers++;
    job->RunStep(&lock);
    if (!lock.owns_lock()) {
        lock.lock();
    }
    job->activeWorkers--;
    if (job->Finished())
        jobListCondition.notify_all();

    return true;
}

void ThreadPool::ForEachThread(std::function<void(void)> func) {
    Barrier *barrier = new Barrier(threads.size() + 1);

    ParallelFor(0, threads.size() + 1, [barrier, &func](int64_t) {
        func();
        if (barrier->Block())
            delete barrier;
    });
}

void ThreadPool::Disable() {
    if (disabled) {
        throw std::runtime_error("ThreadPool already disabled");
    }
    disabled = true;
    if (jobList != nullptr) {
        throw std::runtime_error("Jobs are still running when Disable() is called");
    }
}

void ThreadPool::Reenable() {
    if (!disabled) {
        throw std::runtime_error("ThreadPool is not disabled");
    }
    disabled = false;
}

ThreadPool::~ThreadPool() {
    if (threads.empty())
        return;

    {
        std::lock_guard<std::mutex> lock(mutex);
        shutdownThreads = true;
        jobListCondition.notify_all();
    }

    for (std::thread &thread : threads)
        thread.join();
}

std::string ThreadPool::ToString() const {
    std::string s = "[ ThreadPool threads.size(): " + 
                     std::to_string(threads.size()) + 
                     " shutdownThreads: " + 
                     (shutdownThreads ? "true" : "false") + " ";
    if (mutex.try_lock()) {
        s += "jobList: [ ";
        ParallelJob *job = jobList;
        while (job) {
            s += job->ToString() + " ";
            job = job->next;
        }
        s += "] ";
        mutex.unlock();
    } else
        s += "(job list mutex locked) ";
    return s + "]";
}

bool DoParallelWork() {
    if (!ParallelJob::threadPool) {
        throw std::runtime_error("ParallelJob::threadPool is null");
    }
    // lock should be held when this is called...
    return ParallelJob::threadPool->WorkOrReturn();
}

// ParallelForLoop1D Definition
class ParallelForLoop1D : public ParallelJob {
  public:
    // ParallelForLoop1D Public Methods
    ParallelForLoop1D(int64_t startIndex, int64_t endIndex, int chunkSize,
                      std::function<void(int64_t, int64_t)> func)
        : func(std::move(func)),
          nextIndex(startIndex),
          endIndex(endIndex),
          chunkSize(chunkSize) {}

    bool HaveWork() const { return nextIndex < endIndex; }

    void RunStep(std::unique_lock<std::mutex> *lock);

    std::string ToString() const {
        return "[ ParallelForLoop1D nextIndex: " + std::to_string(nextIndex) + 
               " endIndex: " + std::to_string(endIndex) + 
               " chunkSize: " + std::to_string(chunkSize) + " ]";
    }

  private:
    // ParallelForLoop1D Private Members
    std::function<void(int64_t, int64_t)> func;
    int64_t nextIndex, endIndex;
    int chunkSize;
};

// ParallelForLoop1D Method Definitions
void ParallelForLoop1D::RunStep(std::unique_lock<std::mutex> *lock) {
    // Determine the range of loop iterations to run in this step
    int64_t indexStart = nextIndex;
    int64_t indexEnd = std::min(indexStart + chunkSize, endIndex);
    nextIndex = indexEnd;

    // Remove job from list if all work has been started
    if (!HaveWork())
        threadPool->RemoveFromJobList(this);

    // Release lock and execute loop iterations in _[indexStart, indexEnd)_
    lock->unlock();
    func(indexStart, indexEnd);
}

// Parallel Function Definitions
void ParallelFor(int64_t start, int64_t end, std::function<void(int64_t, int64_t)> func) {
    if (!ParallelJob::threadPool) {
        throw std::runtime_error("ParallelJob::threadPool is null");
    }
    if (start == end)
        return;
    // Compute chunk size for parallel loop
    int64_t chunkSize = std::max<int64_t>(1, (end - start) / (8 * RunningThreads()));

    // Create and enqueue _ParallelForLoop1D_ for this loop
    ParallelForLoop1D loop(start, end, chunkSize, std::move(func));
    std::unique_lock<std::mutex> lock = ParallelJob::threadPool->AddToJobList(&loop);

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished())
        ParallelJob::threadPool->WorkOrWait(&lock, true);
}

///////////////////////////////////////////////////////////////////////////

int AvailableCores() {
    return std::max<int>(1, std::thread::hardware_concurrency());
}

int RunningThreads() {
    return ParallelJob::threadPool ? (1 + ParallelJob::threadPool->size()) : 1;
}

void ParallelInit(int nThreads) {
    if (ParallelJob::threadPool) {
        throw std::runtime_error("ParallelJob::threadPool already initialized");
    }
    if (nThreads <= 0)
        nThreads = AvailableCores();
    ParallelJob::threadPool = new ThreadPool(nThreads);
}

void ParallelCleanup() {
    delete ParallelJob::threadPool;
    ParallelJob::threadPool = nullptr;
}

void ForEachThread(std::function<void(void)> func) {
    if (ParallelJob::threadPool)
        ParallelJob::threadPool->ForEachThread(std::move(func));
}

void DisableThreadPool() {
    if (!ParallelJob::threadPool) {
        throw std::runtime_error("ParallelJob::threadPool is null");
    }
    ParallelJob::threadPool->Disable();
}

void ReenableThreadPool() {
    if (!ParallelJob::threadPool) {
        throw std::runtime_error("ParallelJob::threadPool is null");
    }
    ParallelJob::threadPool->Reenable();
}

}  // namespace gquery
