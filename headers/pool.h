#pragma once

#include "common.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

template <typename R>
class Pool {
public:
  explicit Pool(size_t threadCount = std::thread::hardware_concurrency()) {
    if (threadCount == 0) threadCount = 4;
    m_threads.reserve(threadCount);
    for (size_t i = 0; i < threadCount; ++i) {
      m_threads.emplace_back([this] { this->m_worker(); });
    }
  }

  ~Pool() {
    {
      std::unique_lock<std::mutex> lock(m_queueMutex);
      m_stopping = true;
    }
    m_queueCv.notify_all();
    for (auto &t : m_threads) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  void queue(std::function<R()> task) {
    {
      std::unique_lock<std::mutex> lock(m_queueMutex);
      m_pendingTasks++;
      m_tasks.push_back(std::move(task));
    }
    m_queueCv.notify_one();
  }

  std::vector<R> takeReturnVals() {
    std::unique_lock<std::mutex> lock(m_completionMutex);
    m_completionCv.wait(lock, [this] { return m_pendingTasks.load() == 0; });

    std::lock_guard<std::mutex> rLock(m_resultsMutex);
    std::vector<R> result;
    result.swap(m_results);
    return result;
  }

private:
  void m_worker() {
    while (true) {
      std::function<R()> task;
      {
        std::unique_lock<std::mutex> lock(m_queueMutex);
        m_queueCv.wait(lock, [this] {
          return !m_tasks.empty() || m_stopping;
        });
        if (m_stopping && m_tasks.empty()) {
          return;
        }
        task = std::move(m_tasks.back());
        m_tasks.pop_back();
      }

      R returnVal = task();

      {
        std::lock_guard<std::mutex> rLock(m_resultsMutex);
        m_results.push_back(std::move(returnVal));
      }

      if (--m_pendingTasks == 0) {
        std::lock_guard<std::mutex> lock(m_completionMutex);
        m_completionCv.notify_all();
      }
    }
  }

  std::vector<std::function<R()>> m_tasks;
  std::vector<R> m_results;
  std::vector<std::thread> m_threads;

  std::mutex m_queueMutex;
  std::condition_variable m_queueCv;

  std::mutex m_resultsMutex;

  std::mutex m_completionMutex;
  std::condition_variable m_completionCv;

  std::atomic<int> m_pendingTasks{0};
  bool m_stopping{false};
};
