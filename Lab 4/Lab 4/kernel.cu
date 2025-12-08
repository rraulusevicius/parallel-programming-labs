#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std;

#define MAX_NAME_LENGTH 32       // Max length for student name
#define MAX_RESULT_LENGTH 48     // Max length for result string (NAME-YEARGRADE)
#define THREADS_PER_BLOCK 64     // Multiple of 32 (warp size)
#define NUM_BLOCKS 2             // At least 2 blocks required

struct Student {
    char name[MAX_NAME_LENGTH];
    int year;
    float grade;
};

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << endl;
        exit(1);
    }
}

// Device function: convert to uppercase
__device__ char toUpperCase(char c) {
    if (c >= 'a' && c <= 'z') {
        return c - 32;
    }
    // Did not do Lithuanian letters.
    return c;
}

// Device function: get grade letter based on number (float)
__device__ char getGradeLetter(float grade) {
    if (grade >= 9.0f) return 'A';
    if (grade >= 8.0f) return 'B';
    if (grade >= 7.0f) return 'C';
    if (grade >= 6.0f) return 'D';
    if (grade >= 5.0f) return 'E';
    return  'F';
}

// Device function: check if the name starts with letter P or later in alphabet
__device__ bool passesNameFilter(const char* name) {
    char firstChar = toUpperCase(name[0]);
    return firstChar > 'P';
}

/*
Process student data and generate results.
Each thread processes multiple students (data count > thread count)
Uses atomicAdd to get unique write positions in result array
Uses shared memory to reduce global memory writes
*/
__global__ void ProcessStudents(const Student* students, int studentCount, char* results, int* resultCount, int maxResultLength) {
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    extern __shared__ char sharedResults[];

    int studentsPerThread = studentCount / totalThreads;  
    int remainderStudents = studentCount % totalThreads;

    int startIdx, endIdx;
    if (globalThreadId < remainderStudents) {
        startIdx = globalThreadId * (studentsPerThread + 1);
        endIdx = startIdx + studentsPerThread + 1;
    }
    else {
        startIdx = remainderStudents * (studentsPerThread + 1) + (globalThreadId - remainderStudents) * studentsPerThread;
        endIdx = startIdx + studentsPerThread;
    }

    // Process students assigned to this thread
    for (int i = startIdx; i < endIdx && i < studentCount; i++) {
        const Student* student = &students[i];

        if (passesNameFilter(student->name)) {
            char resultBuffer[MAX_RESULT_LENGTH];
            int pos = 0;

            // Copy name and convert to uppercase
            int nameIdx = 0;
            while (student->name[nameIdx] != '\0' && pos < MAX_RESULT_LENGTH - 5) {
                resultBuffer[pos++] = toUpperCase(student->name[nameIdx++]);
            }

            resultBuffer[pos++] = '-';
            const char digits[] = "0123456789";
            resultBuffer[pos++] = digits[student->year];
            resultBuffer[pos++] = getGradeLetter(student->grade);
            resultBuffer[pos] = '\0';

            int writePos = atomicAdd(resultCount, 1);

            // Write to global results array
            char* destPtr = results + (writePos * maxResultLength);
            for (int j = 0; j <= pos; j++) {
                destPtr[j] = resultBuffer[j];
            }

            for (int j = pos + 1; j < maxResultLength; j++) {
                destPtr[j] = ' ';
            }
        }
    }

}

// Load data from json
vector<Student> loadStudents(const string& filename) {
    vector<Student> students;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return students;
    }
    json data;
    file >> data;
    for (const auto& item : data["students"]) {
        Student student;

        // Copy name
        string name = item["name"].get<string>();
        strncpy(student.name, name.c_str(), MAX_NAME_LENGTH - 1);
        student.name[MAX_NAME_LENGTH - 1] = '\0';

        student.year = item["year"].get<int>();
        student.grade = item["grade"].get<float>();

        students.push_back(student);
    }

    cout << "Loaded " << students.size() << " students from " << filename << endl;
    return students;
}

// Save results to file
void saveResults(const string& filename, const vector<string>& results) {
    ofstream file(filename);
    for (const auto& result : results) {
        file << result << endl;
    }
    file.close();
    cout << "Saved " << results.size() << " results to " << filename << endl;
}

int main(int argc, char* argv[]) {
    cout << "--- Processing students ---\n";

    // Check device
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Failed to get device count");

    if (deviceCount == 0) {
        cerr << "Error: No CUDA-capable GPU found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    checkCudaError(cudaGetDeviceProperties(&prop, 0), "Failed to get device properties");
    
    string inputFile = "students.json";
    string outputFile = "results.txt";

    vector<Student> students = loadStudents(inputFile);
    if (students.empty()) {
        cerr << "Error: No students were found in " << inputFile << endl;
        return 1;
    }

    int studentCount = students.size();
    int totalThreads = NUM_BLOCKS * THREADS_PER_BLOCK;

    cout << "\n -- Current config -- \n";
    cout << "Student count: " << studentCount << endl;
    cout << "Blocks: " << NUM_BLOCKS << endl;
    cout << "Threads per block: " << THREADS_PER_BLOCK << endl;
    cout << "Total threads: " << totalThreads << endl;
    cout << "Data divisible by threads: " << (studentCount % totalThreads == 0 ? "YES" : "NO") << " - required: NO\n";
    cout << "Threads < Data: " << (totalThreads < studentCount ? "YES" : "NO") << " - required: YES\n";

    if (NUM_BLOCKS < 2) {
        cerr << "Error: Must use at least 2 blocks. \n";
        return 1;
    }
    if (THREADS_PER_BLOCK % 32 != 0) {
        cerr << "Error: threads per block must be a multiple of 32. \n";
        return 1;
    }
    if (totalThreads >= studentCount) {
        cerr << "Error: Data count must exceed total thread count. \n";
        return 1;
    }
    
    // Allocate device memory
    Student* d_students;
    char* d_results;
    int* d_resultCount;

    size_t studentsSize = studentCount * sizeof(Student);
    size_t resultsSize = studentCount * MAX_RESULT_LENGTH; // Max possible results

    checkCudaError(cudaMalloc(&d_students, studentsSize), "Failed to allocate device memory for students");
    checkCudaError(cudaMalloc(&d_results, resultsSize), "Failed to allocate device memory for results");
    checkCudaError(cudaMalloc(&d_resultCount, sizeof(int)), "Failed to allocate device memory for result count");

    // Copy data to device
    checkCudaError(cudaMemcpy(d_students, students.data(), studentsSize, cudaMemcpyHostToDevice), "Failed to copy students to device");
    
    //Initialize result count to 0
    int initialCount = 0;
    checkCudaError(cudaMemcpy(d_resultCount, &initialCount, sizeof(int), cudaMemcpyHostToDevice), "Failed to initialize result count");

    // Calculate shared memory size
    size_t sharedMemSize = THREADS_PER_BLOCK * MAX_RESULT_LENGTH;

    cout << "--- Starting GPU work ---\n";

    ProcessStudents << <NUM_BLOCKS, THREADS_PER_BLOCK, sharedMemSize >> > (
        d_students, studentCount, d_results, d_resultCount, MAX_RESULT_LENGTH
        );

    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Wait for GPU to finish
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

    // Copy results back to cpu (host)
    int resultCount;
    checkCudaError(cudaMemcpy(&resultCount, d_resultCount, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy result count from device");

    cout << "Found " << resultCount << " matching students.\n";

    char* h_results = new char[resultsSize];
    checkCudaError(cudaMemcpy(h_results, d_results, resultsSize, cudaMemcpyDeviceToHost), "Failed to copy results from device");

    vector<string> resultStrings;
    for (int i = 0; i < resultCount; i++) {
        char* resultPtr = h_results + (i * MAX_RESULT_LENGTH);
        string result(resultPtr);
        size_t end = result.find_last_not_of(' ');
        if (end != string::npos) {
            result = result.substr(0, end + 1);
        }
        resultStrings.push_back(result);
    }

    // Print results
    cout << "\n --- Results --- \n";
    for (const auto& result : resultStrings) {
        cout << result << endl;
    }

    saveResults(outputFile, resultStrings);

    delete[] h_results;
    checkCudaError(cudaFree(d_students), "Failed to free device memory");
    checkCudaError(cudaFree(d_results), "Failed to free device memory");
    checkCudaError(cudaFree(d_resultCount), "Failed to free device memory");

    cout << "\n --- Done --- \n";

    return 0;
}