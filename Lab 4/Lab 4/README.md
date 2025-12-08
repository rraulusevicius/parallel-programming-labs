# L3 GPU Parallel Programming - Setup Guide for Visual Studio

## Prerequisites

1. **NVIDIA GPU** (You have RTX 3080 - perfect!)
2. **CUDA Toolkit** installed (download from https://developer.nvidia.com/cuda-downloads)
3. **Visual Studio 2019/2022** with C++ workload
4. **nlohmann/json** library for JSON parsing

## Step-by-Step Setup in Visual Studio

### 1. Create CUDA Project
1. Open Visual Studio
2. File → New → Project
3. Search for "CUDA" in templates
4. Select **"CUDA 12.x Runtime"** (or your installed version)
5. Name it `L3_GPU_Students` and create

### 2. Add nlohmann/json Library
**Option A: NuGet (Easiest)**
1. Right-click on project → "Manage NuGet Packages"
2. Search for `nlohmann.json`
3. Install it

**Option B: Single Header (Alternative)**
1. Download `json.hpp` from: https://github.com/nlohmann/json/releases
2. Copy it to your project folder
3. Add to project (Right-click Source Files → Add → Existing Item)

### 3. Add Source Files
1. Delete the default `kernel.cu` file
2. Right-click Source Files → Add → Existing Item
3. Add `main.cu` from this project
4. Also add `students.json` to project folder

### 4. Project Configuration
Right-click project → Properties:

**C/C++ → General:**
- Additional Include Directories: Add path to json.hpp if using Option B

**CUDA C/C++ → Common:**
- Target Machine Platform: 64-bit (for Windows x64)

**CUDA C/C++ → Device:**
- Code Generation: `compute_86,sm_86` (for RTX 3080)
  - RTX 3080 uses Ampere architecture (compute capability 8.6)

### 5. Build and Run
1. Set configuration to **Release x64** (or Debug x64)
2. Build Solution (Ctrl+Shift+B)
3. Make sure `students.json` is in the output directory (or same as .exe)
4. Run (F5)

## Project Structure
```
L3_GPU_Students/
├── main.cu              # Main CUDA source file
├── students.json        # Input data (300 students)
├── results.txt          # Output file (generated)
└── json.hpp             # JSON library (if using Option B)
```

## Understanding the Code

### Key CUDA Concepts Used:

1. **Memory Management**
   ```cpp
   cudaMalloc()      // Allocate GPU memory
   cudaMemcpy()      // Copy data between CPU and GPU
   cudaFree()        // Free GPU memory
   ```

2. **Kernel Launch**
   ```cpp
   processStudents<<<NUM_BLOCKS, THREADS_PER_BLOCK, sharedMemSize>>>(...)
   //              ↑ blocks    ↑ threads/block     ↑ shared memory size
   ```

3. **Thread Indexing**
   ```cpp
   int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
   ```

4. **Atomic Operations**
   ```cpp
   int writePos = atomicAdd(resultCount, 1);  // Thread-safe counter
   ```

5. **Shared Memory** (faster than global memory)
   ```cpp
   extern __shared__ char sharedResults[];
   ```

6. **Synchronization**
   ```cpp
   cudaDeviceSynchronize();  // Wait for GPU to finish
   ```

### Task Requirements Satisfied:

| Requirement | Implementation |
|-------------|----------------|
| ≥250 elements | 300 students in JSON |
| ≥2 blocks | 4 blocks configured |
| Threads per block ÷ 32 | 64 threads (multiple of 32) |
| Total threads < data | 256 threads < 300 students |
| Data NOT ÷ threads | 300 % 256 = 44 (not zero) |
| Minimize global writes | Uses atomicAdd for write position |
| Text result + filter | NAME-YEARGRADE, filter by first letter > 'P' |

## Filter Logic

**Condition:** Name starts with letter after 'P' in alphabet
- Matching: R, S, T, U, V, W, X, Y, Z (and lowercase)
- Examples: **S**onata ✓, **V**acys ✓, **R**obertas ✓
- Non-matching: **A**ntanas ✗, **K**azys ✗, **P**etras ✗

**Grade Mapping:**
| Grade Range | Letter |
|-------------|--------|
| 9.0 - 10.0  | A      |
| 8.0 - 8.99  | B      |
| 7.0 - 7.99  | C      |
| 6.0 - 6.99  | D      |
| 5.0 - 5.99  | E      |
| < 5.0       | F      |

## Expected Output Example
```
SONATA-3A
VACYS-2B
ROBERTAS-3D
VIKTORAS-2B
VYTAS-3C
ZIGMAS-3A
SIMAS-3D
TOMAS-1C
RASA-2B
UGNE-3A
...
```

## Common Issues & Solutions

### "No CUDA-capable device found"
- Ensure NVIDIA drivers are installed
- Check Device Manager for GPU

### "cudaMalloc failed"
- GPU might be out of memory
- Close other GPU applications

### JSON file not found
- Make sure `students.json` is in same directory as .exe
- Or provide full path in code

### Linker errors
- Make sure CUDA toolkit is properly installed
- Check project is set to CUDA project type

## For Lab Demonstration

When demonstrating to your instructor:
1. Show the JSON input file
2. Run the program
3. Show the results.txt output
4. Explain:
   - Block/thread configuration
   - How threads distribute work
   - The atomicAdd for contiguous writes
   - The filter condition

## Further Reading

- CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
