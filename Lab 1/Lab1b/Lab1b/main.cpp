#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <nlohmann/json.hpp>
#include "Product.h"

using json = nlohmann::json;
using std::string;
using std::vector;

// Load products from JSON file
vector<Product> loadProductsFromFile(const string& filename) {
    vector<Product> products;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return products;
    }

    json j;
    file >> j;

    for (const auto& item : j) {
        string name = item.at("name").get<string>();
        double price = item.at("price").get<double>();
        int quantity = item.at("quantity").get<int>();
        products.emplace_back(quantity, price, name);
    }

    return products;
}

// Specialized monitor for storing results in sorted order
// Used to collect filtered results from worker threads
// NOTE: This monitor is NOT thread-safe - uses OpenMP critical sections for synchronization
template <typename T, size_t CAPACITY>
class SortedResultMonitor {
private:
    // Fixed-size array to store sorted results
    std::array<T, CAPACITY> data;

    // Number of results currently stored
    size_t count = 0;

public:
    // Default constructor
    SortedResultMonitor() = default;

    // Add item while maintaining sorted order
    void addSorted(const T& item) 
    {
#pragma omp critical(add_sorted)
        {
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
    }

    // Get all results as a vector (for writing to file)
    std::vector<T> getAllResults() const {
        // Create vector from the first 'count' elements
        return std::vector<T>(data.begin(), data.begin() + count);
    }

    // Get current number of results
    size_t size() const {
        return count;
    }

    // Check if monitor is empty
    bool empty() const {
        return count == 0;
    }
};

// Write results to output file with sums at the end
template <typename T, size_t CAPACITY>
void writeResultsToFile(const SortedResultMonitor<T, CAPACITY>& monitor,
    const string& filename,
    int totalQuantity,
    double totalPrice) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not create output file " << filename << std::endl;
        return;
    }

    // Write table header
    file << std::left
        << std::setw(30) << "Name"
        << std::setw(12) << "Quantity"
        << std::setw(12) << "Price"
        << std::setw(15) << "Risk Score" << "\n";
    file << std::string(69, '-') << "\n";

    // Get results directly from monitor and write each result
    vector<ProductResult> results = monitor.getAllResults();
    for (const auto& result : results) {
        file << std::left
            << std::setw(30) << result.product.name
            << std::setw(12) << result.product.quantity
            << std::setw(12) << std::fixed << std::setprecision(2) << result.product.price
            << std::setw(15) << std::scientific << std::setprecision(5) << result.riskScore << "\n";
    }

    // Write sums at the end
    file << std::string(69, '=') << "\n";
    file << std::left
        << std::setw(30) << "TOTALS (filtered items only):"
        << std::setw(12) << totalQuantity
        << std::setw(12) << std::fixed << std::setprecision(2) << totalPrice
        << "\n";

    file.close();
    std::cout << "Results written to " << filename << "\n";
}

int main() {
    const string inputFile = "IF31_RauluseviciusR_L1_dat_1.json";
    const string outputFile = "IF31_RauluseviciusR_L1_rez.txt";

    auto total_start = std::chrono::high_resolution_clock::now();

    // Load data from file
    std::cout << "-- Loading products from file: " << inputFile << " --\n";
    auto load_start = std::chrono::high_resolution_clock::now();
    vector<Product> products = loadProductsFromFile(inputFile);
    auto load_end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start);

    if (products.empty()) {
        std::cerr << "No products loaded. Exiting.\n";
        return 1;
    }

    std::cout << "Loaded " << products.size() << " products in " << load_duration.count() << " ms\n";

    // Calculate number of threads (2 <= x <= n/4)
    int n = static_cast<int>(products.size());
    int numThreads = std::max(2, std::min(8, n / 4));
    std::cout << "\n-- Using " << numThreads << " threads with OpenMP --\n";

    // Process data in parallel region 
    auto process_start = std::chrono::high_resolution_clock::now();

    // Shared result monitor for all threads
    const size_t MAX_RESULTS = 1000;
    SortedResultMonitor<ProductResult, MAX_RESULTS> resultMonitor;

    // Variables for reduction
    int totalQuantity = 0;
    double totalPrice = 0.0;

    // Single parallel region - all work done here
    // Use reduction for sums instead of critical section
#pragma omp parallel num_threads(numThreads) reduction(+:totalQuantity, totalPrice)
    {
        int threadId = omp_get_thread_num();
        int totalThreads = omp_get_num_threads();

        // Calculate work distribution for this thread and distribute elements as evenly as possible 
        int baseCount = n / totalThreads;           // Base items per thread
        int remainder = n % totalThreads;          // Extra items to distribute

        // First 'extraCount' threads get one extra element
        int start, end;
        if (threadId < remainder) {
            start = threadId * (baseCount + 1);
            end = start + (baseCount + 1);
        }
        else {
            start = remainder * (baseCount + 1) + (threadId - remainder) * baseCount;
            end = start + baseCount;
        }

        // Print thread assignment (still needs critical for console output)
#pragma omp critical(console_output)
        {
            std::cout << "Thread " << threadId << " processing elements " << start
                << " to " << (end - 1) << " (" << (end - start) << " items)\n";
        }

        // Process assigned elements
        for (int i = start; i < end; ++i) {
            // Calculate risk score (complex calculation)
            // This happens independently - other threads can calculate while one adds to monitor
            double riskScore = products[i].calculateRiskScore();

            // Check if passes filter
            if (products[i].isHighRisk(riskScore)) {
                // Add to shared monitor with critical section
                // This is brief - other threads continue calculating

                resultMonitor.addSorted(ProductResult(products[i], riskScore));

            // Add to reduction variables (OpenMP handles this automatically)
            totalQuantity += products[i].quantity;
            totalPrice += products[i].price;
            }
        }
    }
    // End of parallel region - all threads have finished
    // Reduction variables are automatically combined

    auto process_end = std::chrono::high_resolution_clock::now();
    auto process_duration = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start);

    std::cout << "\n-- Processing complete --\n";
    std::cout << "Processing time: " << process_duration.count() << " ms\n";
    std::cout << "Results found: " << resultMonitor.size() << " out of " << products.size() << " products\n";

    // Write results to file directly from monitor
    auto write_start = std::chrono::high_resolution_clock::now();
    writeResultsToFile(resultMonitor, outputFile, totalQuantity, totalPrice);
    auto write_end = std::chrono::high_resolution_clock::now();
    auto write_duration = std::chrono::duration_cast<std::chrono::milliseconds>(write_end - write_start);

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    // Print summary
    std::cout << "\n-- Execution Summary --\n";
    std::cout << "Loading time:    " << load_duration.count() << " ms\n";
    std::cout << "Processing time: " << process_duration.count() << " ms\n";
    std::cout << "Writing time:    " << write_duration.count() << " ms\n";
    std::cout << "Total time:      " << total_duration.count() << " ms\n";
    std::cout << "\n-- Sums (filtered items only) --\n";
    std::cout << "Total Quantity: " << totalQuantity << "\n";
    std::cout << "Total Price: " << std::fixed << std::setprecision(2) << totalPrice << "\n";
    std::cout << "\n-- Done. --\n";

    return 0;
}