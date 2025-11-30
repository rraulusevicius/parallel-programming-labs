#ifndef DATA_MONITOR_H
#define DATA_MONITOR_H

#include <array>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <algorithm>
#include <vector>

// Monitor for thread-safe producer-consumer pattern
// CAPACITY is the maximum number of items that can be stored
template <typename T, size_t CAPACITY>
class DataMonitor {
private:
    std::array<T, CAPACITY> data;

    size_t count = 0;

    size_t front = 0;  
    size_t rear = 0;   

    mutable std::mutex mtx;

    std::condition_variable not_full;   
    std::condition_variable not_empty;  

    bool finished = false;

public:
    DataMonitor() = default;

    void add(const T& item) {
        std::unique_lock<std::mutex> lock(mtx);

        while (count >= CAPACITY) {
            not_full.wait(lock);
        }

        data[rear] = item;

        rear = (rear + 1) % CAPACITY;

        ++count;

        not_empty.notify_one();
    }

    // Rremove an item from the monitor
    // Blocks if the monitor is empty until data becomes available
    // Returns null if finished and empty (no more data coming)
    std::optional<T> get() {
        std::unique_lock<std::mutex> lock(mtx);

        // Wait until there's data or worker is finished
        while (count == 0 && !finished) {
            not_empty.wait(lock);
        }

        if (count == 0 && finished) {
            return std::nullopt;
        }

        T item = data[front];

        front = (front + 1) % CAPACITY;

        --count;

        // Wake up one thread waiting for work
        not_full.notify_one();

        return item;
    }

    // Signal that no more items will be added
    void finish() {
        std::lock_guard<std::mutex> lock(mtx);
        finished = true;

        // Wake up all waiting threads so they can check and exit
        not_empty.notify_all();
    }

    // Check if monitor is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return count == 0;
    }

    // Check if monitor is full
    bool full() const {
        std::lock_guard<std::mutex> lock(mtx);
        return count == CAPACITY;
    }

    // Get current number of items
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return count;
    }
};

// Specialized monitor for storing results in sorted order
// Used to collect filtered results from worker threads
template <typename T, size_t CAPACITY>
class SortedResultMonitor {
private:
    // Fixed-size array to store sorted results
    std::array<T, CAPACITY> data;

    // Number of results currently stored
    size_t count = 0;

    // Mutex for thread-safe access
    mutable std::mutex mtx;

public:
    // Default constructor
    SortedResultMonitor() = default;

    // Add item while maintaining sorted order
    void addSorted(const T& item) {
        std::lock_guard<std::mutex> lock(mtx);

        // Check if we have space
        if (count >= CAPACITY) {
            throw std::runtime_error("Result monitor is full");
        }

        // Find insertion position (linear search from the end)
        size_t pos = count;
        while (pos > 0 && data[pos - 1] > item) {
            data[pos] = data[pos - 1];  // Shift right
            pos--;
        }

        // Insert at found position
        data[pos] = item;
        ++count;
    }

    // Get all results as a vector (for writing to file)
    std::vector<T> getAllResults() const {
        std::lock_guard<std::mutex> lock(mtx);

        // Create vector from the first 'count' elements
        return std::vector<T>(data.begin(), data.begin() + count);
    }

    // Get current number of results
    size_t size() const {
        std::lock_guard<std::mutex> lock(mtx);
        return count;
    }

    // Check if monitor is empty
    bool empty() const {
        std::lock_guard<std::mutex> lock(mtx);
        return count == 0;
    }
};

#endif // DATA_MONITOR_H