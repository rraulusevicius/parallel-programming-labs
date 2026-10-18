#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

#define MAX_NAME_LENGTH 64
#define MAX_RESULT_LENGTH 128
#define THREADS_PER_BLOCK 64
#define NUM_BLOCKS 4
#define RISK_THRESHOLD 100000000.0  // 100 million

struct Product {
    char name[MAX_NAME_LENGTH];
    int quantity;
    double price;
};

struct ProductResult {
    char name[MAX_NAME_LENGTH];
    int quantity;
    double price;
    double riskScore;
};

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << endl;
        exit(1);
    }
}

// Device function: Calculate risk score
__device__ double calculateRiskScore(const Product* product) {
    double score = 0.0;
    const int ITERATIONS = 500000;

    for (int i = 1; i <= ITERATIONS; i++) {
        double baseCalc = (double)(product->quantity) * product->price * (double)i;

        double trigComponent = sin(baseCalc / 1000.0) * cos(baseCalc / 2000.0);
        double logComponent = log(fabs(baseCalc) + 1.0) / log((double)i + 1.0);
        double expComponent = exp(-(double)i / 50000.0) * baseCalc;
        double sqrtComponent = sqrt(fabs(trigComponent * logComponent)) / (double)i;
        double powerComponent = pow(fabs(baseCalc / (double)i), 0.3);

        score += (trigComponent + logComponent + expComponent + sqrtComponent + powerComponent) / (double)i;

        if (i % 1000 == 0) {
            score *= 1.001;
        }

        if (i % 100 == 0) {
            double temp = score;
            for (int j = 0; j < 10; j++) {
                temp = sin(temp) + cos(temp * 0.1);
            }
            score += temp * 0.0001;
        }
    }

    double finalScore = fabs(score);

    double nameHash = 0.0;
    for (int i = 0; i < MAX_NAME_LENGTH && product->name[i] != '\0'; i++) {
        nameHash += (double)(product->name[i]) * 3.14159;
    }
    finalScore += sin(nameHash) * 0.1;

    return finalScore;
}

// Device function: Check if high risk
__device__ bool isHighRisk(double score) {
    return score > RISK_THRESHOLD;
}

// Kernel: Process products and filter high-risk ones
__global__ void ProcessProducts(const Product* products, int productCount, ProductResult* results, int* resultCount, int* totalQuantity, float* totalPrice
) {
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    int productsPerThread = productCount / totalThreads;
    int remainderProducts = productCount % totalThreads;

    int startIdx, endIdx;
    if (globalThreadId < remainderProducts) {
        startIdx = globalThreadId * (productsPerThread + 1);
        endIdx = startIdx + productsPerThread + 1;
    }
    else {
        startIdx = remainderProducts * (productsPerThread + 1) + (globalThreadId - remainderProducts) * productsPerThread;
        endIdx = startIdx + productsPerThread;
    }

    for (int i = startIdx; i < endIdx && i < productCount; i++) {
        const Product* product = &products[i];

        double riskScore = calculateRiskScore(product);

        if (isHighRisk(riskScore)) {
            int writePos = atomicAdd(resultCount, 1);

            ProductResult* result = &results[writePos];

            for (int j = 0; j < MAX_NAME_LENGTH; j++) {
                result->name[j] = product->name[j];
            }
            result->quantity = product->quantity;
            result->price = product->price;
            result->riskScore = riskScore;

            atomicAdd(totalQuantity, product->quantity);
            atomicAdd(totalPrice, (float)product->price);
        }
    }
}

// Load products from JSON file
vector<Product> loadProductsFromFile(const string& filename) {
    vector<Product> products;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return products;
    }

    json data;
    file >> data;

    for (const auto& item : data) {
        Product product;

        string name = item["name"].get<string>();
        strncpy(product.name, name.c_str(), MAX_NAME_LENGTH - 1);
        product.name[MAX_NAME_LENGTH - 1] = '\0';

        product.quantity = item["quantity"].get<int>();
        product.price = item["price"].get<double>();

        products.push_back(product);
    }

    cout << "Loaded " << products.size() << " products from " << filename << endl;
    return products;
}

bool compareByName(const ProductResult& a, const ProductResult& b) {
    return strcmp(a.name, b.name) < 0;
}

// Write results to file
void writeResultsToFile(const string& filename, vector<ProductResult>& results, int totalQuantity, double totalPrice
) {
    sort(results.begin(), results.end(), compareByName);

    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not create output file " << filename << endl;
        return;
    }

    file << left << setw(30) << "Name" << setw(12) << "Quantity" << setw(12) << "Price" << setw(15) << "Risk Score" << "\n";
    file << string(69, '-') << "\n";

    for (const auto& result : results) {
        file << left << setw(30) << result.name << setw(12) << result.quantity << setw(12) << fixed << setprecision(2) << result.price << setw(15) << scientific << setprecision(5) << result.riskScore << "\n";
    }

    file << string(69, '=') << "\n";
    file << left << setw(30) << "TOTALS (filtered items only):" << setw(12) << totalQuantity << setw(12) << fixed << setprecision(2) << totalPrice << "\n";

    file.close();
    cout << "Results written to " << filename << "\n";
}

int main() {
    const string inputFile = "IF31_RauluseviciusR_L1_dat_1.json";
    const string outputFile = "IF31_RauluseviciusR_L4_rez.txt";

    auto totalStart = chrono::high_resolution_clock::now();

    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Failed to get device count");

    if (deviceCount == 0) {
        cerr << "Error: No CUDA device found!" << endl;
        return 1;
    }

    cudaDeviceProp prop;
    checkCudaError(cudaGetDeviceProperties(&prop, 0), "Failed to get device properties");
    cout << "Using GPU: " << prop.name << endl;

    cout << "\n-- Loading products --" << endl;
    auto loadStart = chrono::high_resolution_clock::now();
    vector<Product> products = loadProductsFromFile(inputFile);
    auto loadEnd = chrono::high_resolution_clock::now();
    auto loadDuration = chrono::duration_cast<chrono::milliseconds>(loadEnd - loadStart);

    if (products.empty()) {
        cerr << "No products loaded. Exiting." << endl;
        return 1;
    }

    cout << "Loading time: " << loadDuration.count() << " ms" << endl;

    int productCount = products.size();
    int totalThreads = NUM_BLOCKS * THREADS_PER_BLOCK;

    cout << "\n-- Configuration --" << endl;
    cout << "Product count: " << productCount << endl;
    cout << "Blocks: " << NUM_BLOCKS << endl;
    cout << "Threads per block: " << THREADS_PER_BLOCK << endl;
    cout << "Total threads: " << totalThreads << endl;

    // Allocate device memory
    Product* d_products;
    ProductResult* d_results;
    int* d_resultCount;
    int* d_totalQuantity;
    float* d_totalPrice;

    size_t productsSize = productCount * sizeof(Product);
    size_t resultsSize = productCount * sizeof(ProductResult);  // Max possible results

    checkCudaError(cudaMalloc(&d_products, productsSize), "Failed to allocate device memory for products");
    checkCudaError(cudaMalloc(&d_results, resultsSize), "Failed to allocate device memory for results");
    checkCudaError(cudaMalloc(&d_resultCount, sizeof(int)), "Failed to allocate device memory for result count");
    checkCudaError(cudaMalloc(&d_totalQuantity, sizeof(int)), "Failed to allocate device memory for total quantity");
    checkCudaError(cudaMalloc(&d_totalPrice, sizeof(float)), "Failed to allocate device memory for total price");

    // Copy products to device
    checkCudaError(cudaMemcpy(d_products, products.data(), productsSize, cudaMemcpyHostToDevice), "Failed to copy products to device");

    int zero = 0;
    float zeroFloat = 0.0f;
    checkCudaError(cudaMemcpy(d_resultCount, &zero, sizeof(int), cudaMemcpyHostToDevice), "Failed to initialize result count");
    checkCudaError(cudaMemcpy(d_totalQuantity, &zero, sizeof(int), cudaMemcpyHostToDevice), "Failed to initialize total quantity");
    checkCudaError(cudaMemcpy(d_totalPrice, &zeroFloat, sizeof(float), cudaMemcpyHostToDevice), "Failed to initialize total price");

    cout << "\n-- Starting GPU processing --" << endl;
    auto processStart = chrono::high_resolution_clock::now();

    ProcessProducts << <NUM_BLOCKS, THREADS_PER_BLOCK >> > (d_products, productCount, d_results, d_resultCount, d_totalQuantity, d_totalPrice);

    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Wait for GPU to finish
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

    auto processEnd = chrono::high_resolution_clock::now();
    auto processDuration = chrono::duration_cast<chrono::milliseconds>(processEnd - processStart);

    cout << "Processing time: " << processDuration.count() << " ms" << endl;

    // Copy results back to host
    int resultCount;
    int totalQuantity;
    float totalPriceFloat;

    checkCudaError(cudaMemcpy(&resultCount, d_resultCount, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy result count");
    checkCudaError(cudaMemcpy(&totalQuantity, d_totalQuantity, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy total quantity");
    checkCudaError(cudaMemcpy(&totalPriceFloat, d_totalPrice, sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy total price");

    double totalPrice = (double)totalPriceFloat;

    cout << "Found " << resultCount << " high-risk products out of " << productCount << endl;

    vector<ProductResult> results(resultCount);
    if (resultCount > 0) {
        checkCudaError(cudaMemcpy(results.data(), d_results, resultCount * sizeof(ProductResult), cudaMemcpyDeviceToHost), "Failed to copy results");
    }

    cout << "\n-- Writing results --" << endl;
    auto writeStart = chrono::high_resolution_clock::now();
    writeResultsToFile(outputFile, results, totalQuantity, totalPrice);
    auto writeEnd = chrono::high_resolution_clock::now();
    auto writeDuration = chrono::duration_cast<chrono::milliseconds>(writeEnd - writeStart);

    checkCudaError(cudaFree(d_products), "Failed to free device memory");
    checkCudaError(cudaFree(d_results), "Failed to free device memory");
    checkCudaError(cudaFree(d_resultCount), "Failed to free device memory");
    checkCudaError(cudaFree(d_totalQuantity), "Failed to free device memory");
    checkCudaError(cudaFree(d_totalPrice), "Failed to free device memory");

    auto totalEnd = chrono::high_resolution_clock::now();
    auto totalDuration = chrono::duration_cast<chrono::milliseconds>(totalEnd - totalStart);

    cout << "\n-- Summary --" << endl;
    cout << "Loading time:    " << loadDuration.count() << " ms" << endl;
    cout << "Processing time: " << processDuration.count() << " ms" << endl;
    cout << "Writing time:    " << writeDuration.count() << " ms" << endl;
    cout << "Total time:      " << totalDuration.count() << " ms" << endl;
    cout << "\n-- Done --" << endl;

    return 0;
}
